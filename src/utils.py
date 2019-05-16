import pandas as pd
from .models import UserSong
from sklearn.model_selection import train_test_split
from . import Recommenders


def get_recommendation(user):
    df = pd.DataFrame(list(UserSong.objects.filter(user=user).values('user', 'song', 'times').order_by('-times').distinct()))

    train_data, test_data = train_test_split(df, test_size=0.20, random_state=0)
    pm = Recommenders.popularity_recommender_py()
    pm.create(train_data, 'user', 'song')

    return pm.recommend(user)

