import urllib.parse
import requests
import os
import pandas as pd
import numpy as np
import re
import itertools
import cProfile
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, redirect, request, jsonify, session, url_for, render_template
from flask_cors import CORS
from datetime import datetime

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True,
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
app.config['SECRET_KEY'] = os.urandom(64)
app.config['SESSION_TYPE'] = 'filesystem'
# client = MongoClient('mongodb+srv://root:1234@cluster0.jawqmpu.mongodb.net/')
# db = client["music-app"]
# user_collection = db["users"]


def from_json(json_str):
    return json.loads(json_str)


def ms_to_minutes(ms):
    seconds = ms / 1000
    minutes = seconds / 60
    return round(minutes, 2)


def remove_single_quotes(s):
    return s.replace("'", "")


# Register the custom filter with Jinja2
app.jinja_env.filters['remove_single_quotes'] = remove_single_quotes


app.jinja_env.filters['ms_to_minutes'] = ms_to_minutes

app.jinja_env.filters['from_json'] = from_json
# app.static_url_path = 'static'
# app.static_folder = 'static'
# CORS(app)
CLIENT_ID = 'b4e5e4bb045240f59e63c7ccd0611ea1'
CLIENT_SECRET = 'b3d68e9d44ad4a56a6b2af201c5144a2'
REDIRECT_URI = 'http://localhost:5000/callback'

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1/'
SCOPE = 'user-read-email user-follow-read user-library-read playlist-read-private'
# SCOPE = "user-library-read user-library-modify"

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_handler=cache_handler,
    show_dialog=True
)

sp = Spotify(auth_manager=sp_oauth)


def exchange_code_for_access_token(auth_code):
    payload = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }

    response = requests.post(sp_oauth._token_url, data=payload)

    if response.status_code == 200:
        data = response.json()
        access_token = data.get('access_token')
        return access_token
    else:
        print(f"Error")
        return None


@app.route('/')
def index():
    logged_in = session.get('logged_in')
    current_user = session.get("user_name")
    user_image_url = session.get("user_image_url")
    return render_template('index.html', logged_in=logged_in, current_user=current_user, user_image_url=user_image_url)


@app.route('/login')
def login():
    if not session.get('logged_in'):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    else:
        return redirect(url_for('index'))


@app.route('/callback')
def callback():
    token_info = sp_oauth.get_access_token(request.args["code"])
    session["token_info"] = token_info
    access_token = token_info["access_token"]
    session['access_token'] = access_token
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {access_token}"}

    user_response = requests.get(url, headers=headers)

    if user_response.status_code == 200:
        user_info = user_response.json()
        user_id = user_info["id"]
        session["user_id"] = user_id
        user_name = user_info["display_name"]
        session["user_name"] = user_name
        user_image_url = user_info["images"][0]["url"] if user_info.get(
            "images") else None
        session["user_image_url"] = user_image_url

        user_data = {"user_id": user_id, "access_token": access_token}
        # user_collection.insert_one(user_data)

    if session.get('logged_in'):
        return redirect(url_for('index'))

    session['logged_in'] = True
    return redirect(url_for('index'))


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


def get_user_playlists(access_token):
    url = f"https://api.spotify.com/v1/me/playlists?limit=50"
    headers = {"Authorization": f"Bearer {access_token}"}

    playlist_response = requests.get(url, headers=headers)

    if playlist_response.status_code == 200:
        return playlist_response.json()
    else:
        print(f"Error retrieving playlists: {playlist_response.status_code}")
        return None


def get_playlist_tracks(access_token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {access_token}"}

    tracks_response = requests.get(url, headers=headers)

    if tracks_response.status_code == 200:
        return tracks_response.json()
    else:
        print(f"Error")
        return None
    # return render_template('playlist.html', get_saved_tracks=all_playlist_tracks)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.clear()
    return redirect(url_for('index'))


@app.route('/recommendation')
def recommendation():
    try:
        if 'logged_in' in session and ('access_token' in session or 'user_id' in session):
            access_token = session.get('access_token')
            user_id = session.get('user_id')
            logged_in = session.get('logged_in')

        # Use access_token or user_id for recommendation logic

        # Read data from CSV files
            spotify_df = pd.read_csv('data.csv')
            print(spotify_df)
            data_w_genre = pd.read_csv('data_w_genres.csv')

            # Extract and clean genre data
            data_w_genre['genres_upd'] = data_w_genre['genres'].str.extract(
                r"'([^']*)'", expand=False).str.replace(' ', '_')

            # Extract artist data using extractall and apply list conversion
            spotify_df['artists_upd'] = spotify_df['artists'].str.extractall(
                r"'([^']*)'|\"(.*?)\"")[0].groupby(level=0).apply(list)

            # Concatenate artist and song names
            spotify_df['artists_song'] = spotify_df['artists_upd'].astype(
                str) + spotify_df['name']

            # Sort and remove duplicates
            spotify_df.sort_values(
                ['artists_song', 'release_date'], ascending=False, inplace=True)
            spotify_df.drop_duplicates('artists_song', inplace=True)

            # Merge dataframes and consolidate genre lists
            artists_exploded = spotify_df[[
                'artists_upd', 'id']].explode('artists_upd')
            artists_exploded_enriched = artists_exploded.merge(
                data_w_genre, how='left', left_on='artists_upd', right_on='artists')
            artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull(
            )]
            artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby(
                'id')['genres_upd'].agg(list).reset_index()
            artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(
                lambda x: list(set(itertools.chain.from_iterable(x))) if isinstance(x, list) else [])

            spotify_df = spotify_df.merge(artists_genres_consolidated[[
                                          'id', 'consolidates_genre_lists']], on='id', how='left')

            # Convert release date to year
            spotify_df['year'] = pd.to_datetime(
                spotify_df['release_date'], format='%d/%m/%Y', errors='coerce').dt.year
            float_cols = spotify_df.dtypes[spotify_df.dtypes ==
                                           'float64'].index.values
            ohe_cols = 'popularity'
            # Convert popularity to integer and adjust
            spotify_df['popularity_red'] = (
                spotify_df['popularity'] / 5).astype(int)

            # Handle empty genre lists
            spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(
                lambda d: d if isinstance(d, list) and len(d) > 0 else [])

            # Create feature set
            complete_feature_set = create_feature_set(
                spotify_df, float_cols=float_cols)

            # Get playlist tracks and photos
            tracks = {}
            list_photo = {}
            # for i in sp.current_user_playlists()['items']:
            #     tracks[i['name']] = i['uri'].split(':')[2]
            #     list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']

            # Generate playlist recommendations

            playlist_output_list = get_user_playlists(access_token)

            # track_list = get_user_playlists(access_token)

            id_name = {}
            for i in playlist_output_list['items']:
                playlist_name = i['name']

                playlist_uri_parts = i['uri'].split(':')
                playlist_id = playlist_uri_parts[2]

                id_name[playlist_name] = playlist_id

            recommendations = []

            playlist_random = create_necessary_outputs(
                playlist_name, id_name, spotify_df)

            complete_feature_set_playlist_vector_random, complete_feature_set_nonplaylist_random = generate_playlist_feature(
                complete_feature_set, playlist_random, 1.09)
            random_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_random,
                                                   complete_feature_set_nonplaylist_random)
            recommendations.append(random_top40)
            print(recommendations)
            recommendations_dict = [recommendation.to_dict(
                orient='records') for recommendation in recommendations]

            # Serialize JSON-compatible dictionary to JSON string
            recommendations_json = json.dumps(recommendations_dict)
            print(type(recommendations_dict))
            return render_template('playlist.html', logged_in=logged_in, tracks=recommendations_json,)

    except FileNotFoundError:
        return "CSV file not found", 500

    except pd.errors.EmptyDataError:
        return "Empty CSV file", 500

    except Exception as e:
        return str(e), 500


def ohe_prep(df, column, new_name):
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df


def create_feature_set(df, float_cols):
    # tfidf genre lists
    tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

    tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(
        lambda x: x if isinstance(x, list) else []))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" +
                        i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop=True, inplace=True)

    # explicity_ohe = ohe_prep(df, 'explicit','exp')
    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red', 'pop') * 0.15

    # scale float columns
    floats = df[float_cols].reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(
        floats), columns=floats.columns) * 0.2

    # concanenate all features
    final = pd.concat(
        [genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)

    # add song id
    final['id'] = df['id'].values

    return final


def create_necessary_outputs(playlist_name, id_dic, df):
    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        # print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id']  # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = pd.to_datetime(i['added_at'])

    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values(
        'date_added', ascending=False)
    print(playlist)
    return playlist


def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    complete_feature_set_playlist = complete_feature_set[
        # .drop('id', axis = 1).mean(axis =0)
        complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id', 'date_added']], on='id',
                                                                        how='inner')
    complete_feature_set_nonplaylist = complete_feature_set[
        # .drop('id', axis = 1)
        ~complete_feature_set['id'].isin(playlist_df['id'].values)]

    playlist_feature_set = complete_feature_set_playlist.sort_values(
        'date_added', ascending=False)

    most_recent_date = playlist_feature_set.iloc[0, -1]

    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, 'months_from_recent'] = int(
            (most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)

    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(
        lambda x: weight_factor ** (-x))

    playlist_feature_set_weighted = playlist_feature_set.copy()
    # print(playlist_feature_set_weighted.iloc[:,:-4].columns)
    playlist_feature_set_weighted.update(
        playlist_feature_set_weighted.iloc[:, :-4].mul(playlist_feature_set_weighted.weight, 0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    # playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']
    return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist


def generate_playlist_recos(df, features, nonplaylist_features):
    nonplaylist_features = nonplaylist_features.dropna()
    features = features.dropna()

    # Check if nonplaylist_features or features are empty after dropping missing values
    if nonplaylist_features.empty or features.empty:
        print("Nonplaylist features or features dataframe is empty after dropping missing values.")
        # Return an empty DataFrame if nonplaylist_features or features are empty
        return pd.DataFrame()

    # Check if there are any NaN values remaining in nonplaylist_features or features
    if nonplaylist_features.isnull().values.any() or features.isnull().values.any():
        print("Nonplaylist features or features dataframe still contains NaN values after dropping missing values.")
        return pd.DataFrame()
    # Calculate cosine similarity
    print("Features shape:", features.shape)
    print("Nonplaylist features shape:", nonplaylist_features.shape)

    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis=1).values,
                                               features.values.reshape(1, -1)).flatten()

    print("Nonplaylist_df:", non_playlist_df)

    # Check if non_playlist_df is empty after filtering
    if non_playlist_df.empty:
        print("Nonplaylist_df is empty after filtering.")
        return pd.DataFrame()  # Return an empty DataFrame if non_playlist_df is empty

    # Sort non_playlist_df by similarity and get top 40 recommendations
    non_playlist_df = non_playlist_df.sort_values('sim', ascending=False)

    # Check if non_playlist_df is empty after sorting
    if non_playlist_df.empty:
        print("Nonplaylist_df is empty after sorting.")
        return pd.DataFrame()  # Return an empty DataFrame if non_playlist_df is empty

    # Get top 40 recommendations
    non_playlist_df_top_40 = non_playlist_df.head(40)

    # Add image URLs to the recommendations
    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(
        lambda x: sp.track(x)['album']['images'][1]['url'])

    print(non_playlist_df_top_40)
    return non_playlist_df_top_40


@app.route('/redirect_to_spotify/<track_id>')
def redirect_to_spotify(track_id):
    # Construct the Spotify URI for the track using the track ID
    spotify_uri = f'spotify:track:{track_id}'
    # Redirect the user to the Spotify URI
    return redirect(spotify_uri)


if __name__ == '__main__':
    app.run(debug=True)
