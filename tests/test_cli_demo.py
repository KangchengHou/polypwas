import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "data" / "examples"
HM3_DIR = ROOT / "data" / "ldm" / "ukbEUR_HM3"
DEMO_COORDS = {"chrom": "1", "start": "63063158", "end": "63071830"}


def run_cli(
    *args: str, env_overrides: dict[str, str] | None = None, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    if env_overrides is not None:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "polypwas", *args],
        check=True,
        capture_output=True,
        text=True,
        input=stdin,
        cwd=ROOT,
        env=env,
    )


def write_fake_rscript(script_path: Path) -> None:
    script_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import csv
            import gzip
            import re
            import sys
            from pathlib import Path

            if any("requireNamespace('SBayesRC'" in arg or 'requireNamespace("SBayesRC"' in arg for arg in sys.argv):
                print('SBAYESRC_OK')
                raise SystemExit(0)

            expr = sys.argv[sys.argv.index('-e') + 1]
            ma_match = re.search(r"mafile='([^']+)'", expr)
            out_match = re.search(r"outPrefix='([^']+)'", expr)
            if ma_match is None or out_match is None:
                raise SystemExit('missing mafile/outPrefix')

            ma_path = Path(ma_match.group(1))
            out_path = Path(out_match.group(1))
            print('fake SBayesRC training start')
            print('fake OMP_NUM_THREADS=' + str(__import__('os').environ.get('OMP_NUM_THREADS')))
            rows = list(csv.DictReader(ma_path.open(), delimiter='\t'))
            txt_path = Path(str(out_path) + '.txt')
            txt_path.parent.mkdir(parents=True, exist_ok=True)

            if txt_path.suffix == '.gz':
                handle = gzip.open(txt_path, 'wt')
            else:
                handle = txt_path.open('w')

            with handle as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['BETA'])
                for row in rows:
                    writer.writerow([row['b']])

            print('fake SBayesRC training done')
            """
        )
    )
    script_path.chmod(0o755)


def test_cli_demo_path_outputs_expected_format(tmp_path: Path):
    # Example-data prep assumptions used by the demo flow
    assert (EXAMPLES / "angptl3.ma.gz").exists()
    assert (EXAMPLES / "ldl.ma.gz").exists()
    assert not (EXAMPLES / "angptl3.gene.tsv").exists()

    download_result = run_cli("download-example")
    assert "Example pQTL:" in download_result.stdout
    assert (HM3_DIR / "snp.info").exists()
    assert (HM3_DIR / "ldm.info").exists()
    assert (HM3_DIR / "block1.eigen.bin").exists()
    assert "Demo gene coordinates: chr1:63063158-63071830" in download_result.stdout

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_rscript = tmp_path / "fake_rscript.py"
    write_fake_rscript(fake_rscript)
    config_dir = fake_home / ".polypwas"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(f"Rscript: {fake_rscript}\nplink2: plink2\n")
    cli_env = {"HOME": str(fake_home)}

    weights_path = tmp_path / "demo.wgts.gz"
    train_result = run_cli(
        "train",
        "--pqtl",
        str(EXAMPLES / "angptl3.ma.gz"),
        "--ldm-dir",
        str(HM3_DIR),
        "--threads",
        "10",
        "--out",
        str(weights_path),
        env_overrides=cli_env,
    )
    assert "Preparing SBayesRC input..." in train_result.stdout
    assert "Running SBayesRC training..." in train_result.stdout
    assert "fake SBayesRC training start" in train_result.stdout
    assert "fake OMP_NUM_THREADS=10" in train_result.stdout
    assert f"Wrote demo weights to {weights_path}" in train_result.stdout

    weights_df = pd.read_csv(weights_path, sep="\t")
    assert list(weights_df.columns) == ["BETA"]
    assert len(weights_df) > 0

    gene_info_path = tmp_path / "angptl3.gene.tsv"
    gene_info_path.write_text("ID\tCHROM\tSTART\tEND\nANGPTL3\t1\t63063158\t63071830\n")

    pwas_result = run_cli(
        "assoc",
        "--weights",
        str(weights_path),
        "--gwas",
        str(EXAMPLES / "ldl.ma.gz"),
        "--ldm-dir",
        str(HM3_DIR),
        "--gene-info",
        str(gene_info_path),
        env_overrides=cli_env,
    )

    lines = [line.strip() for line in pwas_result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    assert re.fullmatch(r"CIS_Z=-?\d+\.\d+", lines[0])
    assert re.fullmatch(r"TRANS_Z=-?\d+\.\d+", lines[1])

    pwas_result_with_coords = run_cli(
        "assoc",
        "--weights",
        str(weights_path),
        "--gwas",
        str(EXAMPLES / "ldl.ma.gz"),
        "--ldm-dir",
        str(HM3_DIR),
        "--gene-chr",
        DEMO_COORDS["chrom"],
        "--gene-start",
        DEMO_COORDS["start"],
        "--gene-end",
        DEMO_COORDS["end"],
        env_overrides=cli_env,
    )
    coord_lines = [
        line.strip() for line in pwas_result_with_coords.stdout.splitlines() if line.strip()
    ]
    assert lines == coord_lines


def test_cli_setup_writes_config(tmp_path: Path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    fake_rscript = tmp_path / "fake_rscript.py"
    write_fake_rscript(fake_rscript)
    setup_result = run_cli(
        "setup",
        env_overrides={"HOME": str(fake_home)},
        stdin=f"{fake_rscript}\n",
    )

    config_path = fake_home / ".polypwas" / "config.yaml"
    assert config_path.exists()
    config_text = config_path.read_text()
    assert f"Rscript: {fake_rscript}" in config_text
    assert "plink2: plink2" in config_text
    assert f"Wrote config to {config_path}" in setup_result.stdout
