from transformers import pipeline

def batch_analyze_sentiments():
    """
    Analyzes a pre-defined dictionary of news articles (with dates), grouped by stock ticker,
    and prints a clean, organized report.
    """
    
    # --- 1. Define the News Articles to Analyze ---
    #
    # The data is now a list of dictionaries for each stock.
    # Each dictionary has a 'date' and a 'text' key.
    #
    news_articles = {
        "AAPL": [
            {
                'date': 'July 29, 2024',
                'text': """
                iPhone 17 May Disappoint—But Apple's (AAPL) Foldable Future Has Wall Street Watching. 
                Insider Monkey. On June 26, JPMorgan analyst Samik Chatterjee lowered the price target 
                on the stock to $230.00 (from $240.00) while maintaining an "Overweight" rating.
                """
            },
            {
                'date': 'July 30, 2024 (21 minutes ago)',
                'text': """Jim Cramer Says "Apple Stock Deserves a Premium"
                Insider Monkey
                Apple Inc. (NASDAQ:AAPL) is one of the 21 stocks on Jim Cramer's radar. Cramer showed optimism toward the stock as he believes that the company's CEO and the team deserve the "benefit of the doubt." He commented: "We've had an incredible run over the past few months,
                """
            },
            {
                'date': 'July 29, 2024 (17 hours ago)',
                'text': """
                Apple: Beaten Down, But Ready To Bounce Back
                Seeking Alpha
                Discover why Apple Inc.'s resilient financials, ecosystem strength, and undervalued stock make it a Strong Buy despite headwinds. Click for my AAPL update.
                """
            }
        ],
        "TSLA": [
            {
                'date': 'July 28, 2024',
                'text': """
                Elon Musk Grabs Control of Tesla's Europe Sales as Painful Plunge Continues. 
                Can That Save TSLA Stock? Investors are glued to Tesla (TSLA) at writing following 
                reports that its billionaire chief executive, Elon Musk, will personally oversee European sales.
                """
            },
            {
                'date': 'July 29, 2024',
                'text': """
                Why Tesla (TSLA) Shares Are Trading Lower Today. StockStory.org. 
                Shares of electric vehicle pioneer Tesla (NASDAQ:TSLA) fell 5.5% in the afternoon session 
                after the company faced mounting concerns over its upcoming second-quarter delivery report.
                """
            },
            {
                'date': 'July 25, 2024',
                'text': """
                Wall Street Price Prediction: Tesla's Share Price Forecast for 2025.
                After soaring in 2023 and 2024, shares of Tesla (NASDAQ:TSLA) were battered 
                throughout Q1 2025. And while the stock performed marginally better in Q2, 
                the largest U.S. EV-maker slid into Q3.
                """
            }
        ],
        "GOOG": [
            {
                'date': 'July 19, 2024',
                'text': """
                Google hit with $314 million US verdict in cellular data class action.
                """
            },
            {
                'date': 'July 22, 2024',
                'text': """
                Google's data center energy use doubled in 4 years. TechCrunch. 
                Google has pledged to use only carbon-free sources of electricity to power 
                its operations, a task made more challenging by its breakneck pace of data center growth.
                """
            },
            {
                'date': 'July 26, 2024',
                'text': """
                Massive leak reveals Google Pixel 10 Pro specs — plus, Pixel 10 Pro XL updates.
                As for the new Tensor G5 chip, Google is reportedly turning to TSMC to manufacture 
                the new chip on its 3nm process. This should introduce much improved performance and power efficiency.
                """
            }
        ],
        "NVDA": [
            {
                'date': 'July 30, 2024',
                'text': """
                Nvidia's stock continues its meteoric rise, with analysts raising price targets across 
                the board, citing insatiable demand for its AI GPUs.
                """
            },
            {
                'date': 'July 15, 2024',
                'text': """
                Nvidia unveiled its next-generation Blackwell platform, promising unprecedented 
                performance gains for AI and high-performance computing.
                """
            },
            {
                'date': 'July 24, 2024',
                'text': """
                Competition is heating up in the AI chip market, with rivals like AMD and Intel 
                looking to challenge Nvidia's dominance. This could pressure margins in the long term.
                """
            }
        ],
        "TSM": [
            {
                'date': 'June 10, 2024',
                'text': """
                TSMC, Apple and Nvidia supplier, posts 60% jump in May revenue amid AI boom. 
                The results reinforce expectations that the global AI boom is continuing to fuel 
                demand for the high-end chips that TSMC produces.
                """
            },
            {
                'date': 'July 29, 2024',
                'text': """
                Geopolitical tensions remain a significant risk for TSMC, with concerns over a potential
                conflict in the Taiwan Strait weighing on investor sentiment.
                """
            },
            {
                'date': 'July 18, 2024',
                'text': """
                TSMC is on track to start producing 2nm chips in 2025, a move that could 
                solidify its lead in the semiconductor industry.
                """
            }
        ],
        "AVGO": [
            {
                'date': 'July 12, 2024',
                'text': """
                Broadcom's acquisition of VMware is already paying dividends, with the company 
                reporting strong synergistic growth and raising its full-year revenue forecast.
                """
            },
            {
                'date': 'July 23, 2024',
                'text': """
                Regulatory scrutiny over Broadcom's business practices continues, with European 
                regulators opening a new probe into its licensing agreements after the VMware merger.
                """
            }
        ],
        "PLTR": [
            {
                'date': 'July 29, 2024',
                'text': """
                Palantir wins a new multi-million dollar contract with the US Army to deploy 
                its AI-powered software, expanding its footprint in the defense sector.
                """
            },
            {
                'date': 'May 06, 2024',
                'text': """
                Palantir's latest earnings report showed strong revenue growth but missed analyst 
                expectations on profitability, leading to a volatile trading session.
                """
            }
        ],
        "RTX": [
            {
                'date': 'July 30, 2024',
                'text': """
                RTX's Pratt & Whitney division received a massive order for its fuel-efficient 
                GTF engines from a major airline, signaling a strong recovery in the commercial aerospace sector.
                """
            },
            {
                'date': 'July 25, 2023', # Note: Old date for historical context
                'text': """
                RTX announced a significant charge related to a powder metal defect in its engine parts, 
                which will require costly inspections and repairs, impacting short-term profitability.
                """
            }
        ],
        "LHX": [
            {
                'date': 'July 15, 2024',
                'text': """
                L3Harris Technologies secured a major contract from the U.S. Space Force for 
                developing advanced satellite communication systems, boosting its order backlog.
                """
            },
            {
                'date': 'July 23, 2024',
                'text': """
                The company reported that supply chain disruptions have led to minor delays 
                in some of its key defense programs, impacting quarterly revenue projections.
                """
            }
        ],
        "ASTS": [
            {
                'date': 'April 20, 2024',
                'text': """
                AST SpaceMobile successfully completed a test call between a standard smartphone and its satellite,
                a key milestone in its quest to build a space-based cellular broadband network.
                """
            },
            {
                'date': 'July 30, 2024',
                'text': """
                AST SpaceMobile is a pre-revenue company facing significant execution risk 
                and requiring substantial future capital to deploy its full satellite constellation.
                """
            }
        ],
        "RKLB": [
            {
                'date': 'July 28, 2024',
                'text': """
                Rocket Lab successfully launched another Electron rocket, continuing its high-cadence 
                launch schedule and demonstrating reliability for its customers.
                """
            },
            {
                'date': 'July 10, 2024',
                'text': """
                Development of the larger, reusable Neutron rocket is progressing but faces a competitive 
                landscape and a challenging timeline, which could push back revenue expectations.
                """
            }
        ]
    }

    # --- 2. Load the Sentiment Analysis Model ---
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    print(f"Loading model '{model_name}'...\n")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")
        return

    # --- 3. Analyze Articles and Print the Report ---
    print("="*60)
    print("               SENTIMENT ANALYSIS REPORT")
    print("="*60)

    # Loop through each ticker in our dictionary
    for ticker, articles in news_articles.items():
        print(f"\n\n--- Analysis for {ticker} ---")
        if not articles:
            print("No articles to analyze.")
            continue

        # Loop through each article dictionary for the current ticker
        for i, article_data in enumerate(articles, 1):
            article_date = article_data['date']
            article_text = article_data['text']
            
            # Clean up the article text (remove extra whitespace)
            clean_article = " ".join(article_text.strip().split())
            
            try:
                results = sentiment_pipeline(clean_article)
                result = results[0]
                label = result['label'].capitalize()
                score = result['score']
                
                print(f"\n[ Article {i} ]")
                print(f"Date: {article_date}") # Display the date
                print(f"Text: {clean_article[:100]}...") # Print the first 100 chars
                print(f"  -> Sentiment: {label} (Confidence: {score:.2%})")

            except Exception as e:
                print(f"\n[ Article {i} ]")
                print(f"Could not analyze article. Error: {e}")
        
        print("-" * 25)

if __name__ == "__main__":
    batch_analyze_sentiments()