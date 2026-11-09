from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ------------------------- LOAD DATA -------------------------
def load_data():
    # Resolve the CSV beside app.py, even if launched from another folder.
    csv_path = Path(__file__).with_name("recent_movies_2024_2026.csv")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("CSV file not found. Place recent_movies_2024_2026.csv beside app.py.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please add the movie dataset.")
        st.stop()
    except Exception as error:
        st.error(f"Unable to load the movie dataset: {error}")
        st.stop()

    if data.empty:
        st.error("The CSV has column headings but no movie records.")
        st.stop()

    required_columns = [
        "title", "release_year", "release_month", "release_day", "genres",
        "rating_out_of_10", "vote_count", "runtime_minutes", "box_office_usd",
        "budget_usd", "director", "screenwriter", "production_company",
        "content_rating",
    ]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        st.error("The CSV is missing required columns: " + ", ".join(missing_columns))
        st.stop()
    return data


# ------------------------- CLEAN DATA -------------------------
def clean_data(df):
    # Remove repeated records without changing the original dataframe.
    df = df.drop_duplicates().copy()

    # Strip spaces from text while preserving missing values.
    text_columns = [
        "title", "genres", "director", "screenwriter",
        "production_company", "content_rating",
    ]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Keep movies with missing credits and label those credits as Unknown.
    for column in ["director", "screenwriter"]:
        df[column] = df[column].replace("", pd.NA).fillna("Unknown")

    # Convert numeric fields; invalid entries become missing values.
    numeric_columns = [
        "release_year", "release_day", "rating_out_of_10", "vote_count",
        "runtime_minutes", "box_office_usd", "budget_usd",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Retain missing release days and all other incomplete movie records.
    # Remove missing values only from the data needed for individual charts.

    # Calculate profit as box-office revenue minus the production budget.
    df["profit_usd"] = df["box_office_usd"] - df["budget_usd"]
    return df


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    if value < 0:
        return f"-${abs(value) / 1_000_000:,.1f}M"
    return f"${value / 1_000_000:,.1f}M"


def get_unique_genres(df):
    genres = set()
    for value in df["genres"].dropna():
        for genre in value.split(","):
            if genre.strip():
                genres.add(genre.strip())
    return sorted(genres)


def apply_filters(df, search, genre, years, rating_range, content_ratings):
    filtered_df = df.copy()
    if search.strip():
        filtered_df = filtered_df[
            filtered_df["title"].str.contains(
                search.strip(), case=False, na=False, regex=False
            )
        ]
    if genre != "All Genres":
        # Match a complete genre in each comma-separated list.
        genre_matches = filtered_df["genres"].fillna("").apply(
            lambda value: genre in [item.strip() for item in value.split(",")]
        )
        filtered_df = filtered_df.loc[genre_matches.astype(bool)]
    filtered_df = filtered_df[filtered_df["release_year"].isin(years)]
    filtered_df = filtered_df[filtered_df["content_rating"].isin(content_ratings)]
    if rating_range is not None:
        # The full range retains unrated movies; a narrowed range requires a rating.
        full_range = (df["rating_out_of_10"].min(), df["rating_out_of_10"].max())
        if tuple(rating_range) != full_range:
            filtered_df = filtered_df[
                filtered_df["rating_out_of_10"].between(*rating_range)
            ]
    return filtered_df


# ------------------------- VISUALIZATIONS -------------------------
def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_rating_distribution(df):
    ratings = df["rating_out_of_10"].dropna()
    if ratings.empty:
        st.info("No valid ratings are available for the distribution chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratings, bins=10, color="steelblue", edgecolor="white")
    ax.set(title="Movie Rating Distribution", xlabel="Rating", ylabel="Number of Movies")
    show_plot(fig)


def plot_genres(df):
    # Explode a temporary Series, keeping one row per movie in the main dataframe.
    genres = df["genres"].dropna().str.split(",").explode().str.strip()
    counts = genres[genres != ""].value_counts().head(10).sort_values()
    if counts.empty:
        st.info("No genres are available for the genre chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(counts.index, counts.values, color="steelblue")
    ax.set(title="Top Genres by Number of Movies", xlabel="Number of Movies", ylabel="Genre")
    show_plot(fig)


def plot_release_year(df):
    years = df["release_year"].dropna()
    if years.empty:
        st.info("No release years are available for the year chart.")
        return
    counts = years.value_counts().reindex([2024, 2025, 2026], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set(title="Movies Released by Year", xlabel="Release Year", ylabel="Number of Movies")
    show_plot(fig)


def plot_budget_vs_box_office(df):
    plot_df = df.dropna(subset=["budget_usd", "box_office_usd"])
    if plot_df.empty:
        st.info("No valid budget and revenue pairs are available for the scatter plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        plot_df["budget_usd"] / 1_000_000,
        plot_df["box_office_usd"] / 1_000_000,
        alpha=0.6, color="steelblue",
    )
    ax.set(
        title="Budget vs Box Office Revenue",
        xlabel="Budget (USD millions)", ylabel="Box Office (USD millions)",
    )
    show_plot(fig)


def plot_top_movies(df, ranking):
    ranking_columns = {
        "Rating": "rating_out_of_10", "Box Office": "box_office_usd",
        "Vote Count": "vote_count", "Profit": "profit_usd",
    }
    column = ranking_columns[ranking]
    top_movies = df.dropna(subset=["title", column]).nlargest(10, column)
    top_movies = top_movies.sort_values(column)
    if top_movies.empty:
        st.info(f"No valid movies are available to rank by {ranking.lower()}.")
        return
    values = top_movies[column]
    label = ranking
    if ranking in ["Box Office", "Profit"]:
        values = values / 1_000_000
        label += " (USD millions)"
    fig, ax = plt.subplots(figsize=(12, 6))
    # Separate positions also allow different movies with the same title.
    positions = range(len(top_movies))
    ax.barh(positions, values, color="steelblue")
    ax.set_yticks(list(positions), labels=top_movies["title"])
    ax.set(title=f"Top 10 Movies by {ranking}", xlabel=label, ylabel="Movie")
    show_plot(fig)


def main():
    st.set_page_config(
        page_title="Movie Box Office & Audience Analytics Platform",
        page_icon="🎬",
        layout="wide",
    )
    st.title("Movie Box Office & Audience Analytics Platform")
    st.caption(
        "Analyzes recent movie box-office performance and audience trends using "
        "ratings, genres, budgets, revenue, runtime, vote count, and release information."
    )
    original_df = load_data()
    df = clean_data(original_df)

    # ------------------------- SIDEBAR FILTERS -------------------------
    with st.sidebar:
        st.title("Filter Movies")
        search = st.text_input("Movie Title Search")
        genre = st.selectbox("Genre", ["All Genres"] + get_unique_genres(df))
        years = st.multiselect("Release Year", [2024, 2025, 2026], default=[2024, 2025, 2026])
        ratings = df["rating_out_of_10"].dropna()
        rating_range = None
        if ratings.empty:
            st.info("No valid ratings are available for the rating filter.")
        elif ratings.min() == ratings.max():
            st.info(f"All available ratings are {ratings.min():.1f}/10.")
        else:
            minimum, maximum = float(ratings.min()), float(ratings.max())
            rating_range = st.slider(
                "Rating Range", minimum, maximum, (minimum, maximum), step=0.1
            )
        content_options = sorted(df["content_rating"].dropna().unique().tolist())
        content_ratings = st.multiselect("Content Rating", content_options, default=content_options)

    filtered_df = apply_filters(df, search, genre, years, rating_range, content_ratings)

    # ------------------------- KPI METRICS -------------------------
    columns = st.columns(4)
    average_rating = filtered_df["rating_out_of_10"].mean()
    columns[0].metric("Total Movies", len(filtered_df))
    columns[1].metric("Average Rating", "N/A" if pd.isna(average_rating) else f"{average_rating:.1f}/10")
    columns[2].metric("Total Box Office", format_currency(filtered_df["box_office_usd"].sum()))
    columns[3].metric("Total Profit", format_currency(filtered_df["profit_usd"].sum()))

    overview, records, visualizations = st.tabs(["Overview", "Movie Records", "Visualizations"])
    with overview:
        st.info(f"Showing {len(filtered_df)} movies after applying the selected filters.")
        if filtered_df.empty:
            st.warning("No movies match the selected filters.")
        else:
            highlights = [
                ("Highest Rated Movie", "rating_out_of_10"),
                ("Highest Grossing Movie", "box_office_usd"),
                ("Most Voted Movie", "vote_count"),
            ]
            for panel, (heading, column) in zip(st.columns(3), highlights):
                with panel:
                    st.subheader(heading)
                    valid_movies = filtered_df.dropna(subset=[column])
                    if valid_movies.empty:
                        st.info("No valid values are available.")
                        continue
                    movie = valid_movies.loc[valid_movies[column].idxmax()]
                    if column == "rating_out_of_10":
                        value = f"{movie[column]:.1f}/10"
                    elif column == "vote_count":
                        value = f"{movie[column]:,.0f} votes"
                    else:
                        value = format_currency(movie[column])
                    st.write(f"{movie['title']} — {value}")

    with records:
        if filtered_df.empty:
            st.warning("No movies match the selected filters.")
        else:
            record_columns = [
                "title", "release_year", "release_month", "genres", "rating_out_of_10",
                "vote_count", "runtime_minutes", "budget_usd", "box_office_usd",
                "profit_usd", "director", "content_rating",
            ]
            st.dataframe(filtered_df[record_columns], use_container_width=True, hide_index=True)

    with visualizations:
        if filtered_df.empty:
            st.warning("No movies match the selected filters.")
        else:
            left, right = st.columns(2)
            with left:
                plot_rating_distribution(filtered_df)
                plot_release_year(filtered_df)
            with right:
                plot_genres(filtered_df)
                plot_budget_vs_box_office(filtered_df)
            ranking = st.selectbox("Rank Top 10 Movies By", ["Rating", "Box Office", "Vote Count", "Profit"])
            plot_top_movies(filtered_df, ranking)

    # ------------------------- DATASET INFORMATION -------------------------
    with st.expander("Dataset Information"):
        st.write(f"Original rows: {len(original_df)} | Cleaned rows: {len(df)}")
        st.write(f"Number of columns: {len(original_df.columns)} original, {len(df.columns)} after adding profit_usd")
        st.write(f"Number of duplicate rows removed: {len(original_df) - len(df)}")
        missing_columns = ["release_day", "director", "screenwriter"]
        st.caption("Missing values before and after cleaning (entire dataset)")
        st.dataframe(pd.DataFrame({
            "Before cleaning": original_df[missing_columns].isna().sum(),
            "After cleaning": df[missing_columns].isna().sum(),
        }), use_container_width=True)
        st.caption("Cleaned column data types")
        st.dataframe(df.dtypes.astype(str).to_frame("Data type"), use_container_width=True)


if __name__ == "__main__":
    main()
