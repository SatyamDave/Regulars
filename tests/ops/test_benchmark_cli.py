from __future__ import annotations

import pytest

from regulars.ops.benchmark_cli import build_parser, main


def test_cli_assembly_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["assembly", "--sizes", "50,200", "--iterations", "50"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n_users" in out
    assert "50" in out and "200" in out


def test_cli_rejects_empty_sizes() -> None:
    with pytest.raises(ValueError, match="--sizes"):
        main(["assembly", "--sizes", ""])


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["assembly"])
    assert args.command == "assembly"
    assert args.sizes == "50,200,500,1000,2000"
    assert args.iterations == 200
    assert args.avg_candidates_per_user == 200
    assert args.seed == 0
