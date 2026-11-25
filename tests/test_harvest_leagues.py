from typer.testing import CliRunner

from dpow import NEW_LEAGUES, wndx
from dpow.cli import app


def test_wind_covers_the_new_leagues():
    assert set(NEW_LEAGUES) <= set(wndx.LEAGUES)


def test_nbhd_skips_a_league_its_source_does_not_cover(monkeypatch):
    # hsp is US-only: the command must stop before it pulls or marks anything
    import dpow.cli as cli

    monkeypatch.setattr(cli, "use_pb_data", lambda: (_ for _ in ()).throw(AssertionError("ran on")))
    res = CliRunner().invoke(app, ["nbhd", "--league", "japan", "--source", "hsp"])
    assert res.exit_code == 0, res.output
