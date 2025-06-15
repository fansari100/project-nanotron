import datetime as dt

import pytest

pytest.importorskip("respx")

import respx

from nanotron_data.connectors.news.edgar import EdgarClient
from nanotron_data.connectors.news.newsapi import NewsApiClient
from nanotron_data.connectors.news.reddit import RedditClient


@pytest.mark.asyncio
async def test_newsapi_normalizes_articles():
    client = NewsApiClient(api_key="test")
    payload = {
        "articles": [
            {
                "publishedAt": "2024-01-15T12:34:00Z",
                "source": {"name": "Reuters"},
                "title": "Headline A",
                "description": "desc A",
                "url": "http://example.com/a",
            }
        ]
    }
    with respx.mock(base_url="https://newsapi.org") as mock:
        mock.get(url__regex=r"/v2/everything.*").respond(200, json=payload)
        df = await client.search(
            "AAPL",
            dt.datetime(2024, 1, 14, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 16, tzinfo=dt.timezone.utc),
        )
    assert df.iloc[0]["source"] == "Reuters"
    assert df.iloc[0]["title"] == "Headline A"


@pytest.mark.asyncio
async def test_edgar_recent_filings_filters_form_types():
    client = EdgarClient(user_agent="test")
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000-1", "0000-2", "0000-3"],
                "form": ["10-K", "DEF 14A", "10-Q"],
                "filingDate": ["2024-01-15", "2024-02-01", "2024-03-15"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "primaryDocDescription": ["10-K", "Proxy", "10-Q"],
            }
        }
    }
    with respx.mock(base_url="https://data.sec.gov") as mock:
        mock.get(url__regex=r"/submissions/CIK\d+\.json").respond(200, json=payload)
        df = await client.recent_filings(320193)  # AAPL CIK
    assert set(df["form"]) == {"10-K", "10-Q"}
    assert len(df) == 2


@pytest.mark.asyncio
async def test_reddit_hot_returns_normalized_rows():
    client = RedditClient()
    payload = {
        "data": {
            "children": [
                {"data": {
                    "id": "a", "title": "TSLA going parabolic",
                    "created_utc": 1700000000, "score": 100, "num_comments": 25,
                    "url": "http://r/a",
                }},
                {"data": {
                    "id": "b", "title": "GME squeeze take 2",
                    "created_utc": 1700001000, "score": 50, "num_comments": 5,
                    "url": "http://r/b",
                }},
            ]
        }
    }
    with respx.mock(base_url="https://www.reddit.com") as mock:
        mock.get(url__regex=r"/r/wallstreetbets/hot\.json.*").respond(200, json=payload)
        df = await client.hot("wallstreetbets")
    assert set(df["id"]) == {"a", "b"}
    assert df.iloc[0]["score"] in (50, 100)
