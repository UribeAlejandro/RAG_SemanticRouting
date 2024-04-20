from langchain_community.tools.tavily_search import TavilySearchResults


def search_tool() -> TavilySearchResults:
    """Returns a search tool for the project.

    Returns
    -------
    TavilySearchResults
        Toll that uses Tavily Search API and gets back search results
    """
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool
