import pandas as pd

from dpow import NEW_LEAGUES
from dpow.models import stacker
from dpow.predict import bundle_names


def test_new_leagues_use_the_nh_model():
    for lg in NEW_LEAGUES:
        assert bundle_names(lg) == ["nh"]
    assert bundle_names("stations") == ["nh"]
    assert bundle_names("resorts-na") == ["resorts-na", "nh"]


def test_l2_league_codes_of_the_old_leagues_are_unchanged():
    df = pd.DataFrame({"league": ["alps", "observers", "stations", *NEW_LEAGUES]})
    codes = stacker.l2_matrix(df)["league_code"].tolist()
    assert codes[:3] == [0.0, 1.0, 2.0]
    assert codes[3:] == [3.0, 4.0, 5.0]
