import time

import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
import json
from dotenv import load_dotenv
import os
import logging
import base64
from langchain_core.tools import BaseTool
from difflib import get_close_matches
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
llm= ChatOpenAI(model="gpt-4o")

# Neo4j database setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jDatabase:
    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def search_products(self, query):
        print(f"QUERY IWTHOUT P- {query}")
        cypher_query = """
        MATCH (p:Product)
        WHERE p.name =~ $query OR p.description =~ $query
        RETURN p.name, p.price, p.description, p.image_link, p.availability, p.brand
        LIMIT 5
        """
        return self.run_query(cypher_query, {"query": f"(?i).*{query}.*"})

    def get_product_details(self, product_name):
        cypher_query = """
           MATCH (p:Product {name: $name})
           RETURN p.name, p.price, p.description, p.image_link, p.availability, p.brand
           """
        results = self.run_query(cypher_query, {"name": product_name})
        return results[0] if results else None

    def check_recipe_exists(self, recipe):
        cypher_query = """
        MATCH (p:Product)-[:HAS_RECIPE]->(r:Recipe)
        WHERE toLower(r.name) CONTAINS toLower($recipe)
        RETURN p.name AS product, r.name AS recipe
        LIMIT 1
        """
        results = self.run_query(cypher_query, {"recipe": recipe.replace("recipe", "").strip()})
        return results[0] if results else None

db = Neo4jDatabase()

# Tools
import re

import re


class ProductSearchTool(BaseTool):
    name = "ProductSearch"
    description = "Search for products in the store. Input should be a search query string, optionally followed by a Brand specification."

    def _run(self, query: str) -> str:
        print(f"QUERY IS: {query}")
        if query is None:
            query=""
        parts = query.split('|')
        search_query = parts[0].strip()
        brand = parts[1].strip() if len(parts) > 1 else None

        cypher_query = """
        MATCH (p:Product)
        WHERE (p.name =~ $name_query OR p.description =~ $name_query)
        """

        params = {
            "name_query": f"(?i).*{search_query}.*"
        }

        if brand:
            cypher_query += """
            AND toLower(p.brand) = toLower($brand)
            """
            params["brand"] = brand

        cypher_query += """
        RETURN p.name, p.price, p.description, p.image_link, p.availability, p.quantity, p.brand
        LIMIT 5
        """

        print(f"CYPHER QUERY: {cypher_query}")
        print(f"PARAMS: {params}")

        results = db.run_query(cypher_query, params)
        print(f"QUERY RESULTS: {results}")
        return json.dumps([dict(r) for r in results])


    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

def fuzzy_match_recipe(query):
    cypher_query = """
    MATCH (r:Recipe)
    RETURN r.name AS recipe_name
    """
    results = db.run_query(cypher_query)
    recipe_names = [record['recipe_name'] for record in results]
    matches = get_close_matches(query, recipe_names, n=1, cutoff=0.6)
    return matches[0] if matches else None



class ProductDetailsTool(BaseTool):
    name = "ProductDetails"
    description = "Get detailed information about a specific product. Input should be the exact product name."

    def _run(self, product_name: str) -> str:
        details = db.get_product_details(product_name)
        return json.dumps(dict(details)) if details else "Product not found."

    def _arun(self, product_name: str) -> str:
        raise NotImplementedError("This tool does not support async")

class RecipeCheckTool(BaseTool):
    name = "RecipeSearch"
    description = "Search for a recipe by name, using fuzzy matching. Returns the recipe name and its main ingredient (product)."

    def _run(self, recipe_name: str) -> str:
        logger.debug(f"Searching for recipe: {recipe_name}")
        matched_recipe = fuzzy_match_recipe(recipe_name)
        if matched_recipe:
            cypher_query = """
            MATCH (p:Product)-[:HAS_RECIPE]->(r:Recipe)
            WHERE r.name = $recipe_name
            RETURN p.name AS product, r.name AS recipe
            LIMIT 1
            """
            result = db.run_query(cypher_query, {"recipe_name": matched_recipe})
            if result:
                return json.dumps({"found": True, "product": result[0]['product'], "recipe": result[0]['recipe'], "original_query": recipe_name})
        return json.dumps({"found": False, "original_query": recipe_name})

    async def _arun(self, recipe_name: str) -> str:
        return self._run(recipe_name)

product_search_tool = ProductSearchTool()
product_details_tool = ProductDetailsTool()
recipe_check_tool = RecipeCheckTool()

# Agents
query_classifier = Agent(
    role='Query Classifier',
    goal='Classify user queries and extract relevant information',
    backstory="""You are an expert in understanding user intents in e-commerce contexts. You are a personal shopping assistant for a user.
    You can identify various types of queries and extract relevant information such as recipe names or product names.""",
    tools=[],
    verbose=True,
    llm=llm
)

product_searcher = Agent(
    role='Product Searcher',
    goal='Find products based on user queries',
    backstory='You are an expert in searching and presenting product information.',
    tools=[product_search_tool, product_details_tool],
    verbose=True,
    llm=llm
)

recipe_name_extractor = Agent(
    role='Recipe Name Extractor',
    goal='Extract the exact recipe name from user queries',
    backstory="""You are an expert in understanding recipe queries. 
       Your task is to extract only the recipe name from user inputs, ignoring any other words.""",
    tools=[],
    verbose=True,
    llm=llm
)

response_generator = Agent(
    role='Response Generator',
    goal='Generate helpful and coherent responses to user queries',
    backstory="""You are an expert in customer service and communication. You are a personal shopping assistant for a user.
    You format responses in a visually appealing way using Markdown.
    For product listings, you include product images using HTML img tags.
    You ensure that all relevant information is presented clearly and concisely and in a personal way.""",
    tools=[],
    verbose=True,
    llm=llm
)

# Crew
ecommerce_crew = Crew(
    agents=[query_classifier, product_searcher, recipe_name_extractor, response_generator],
    tasks=[],
    verbose=2
)


import json
import logging
import re

logger = logging.getLogger(__name__)

custom_css = """
<style>
    .stButton > button {
        border: 2px solid #FF0000;
        border-radius: 5px;
        color: #4CAF50;
        background-color: white;
        padding: 10px 24px;
        color: #FF0000;
        transition-duration: 0.4s;
    }
    .stButton > button:hover {
        background-color: #4CAF50;
        color: white;
    }
    
     .bordered-image img {
        border: 2px solid #FF0000;
        border-radius: 3px;
        padding: 5px;
        width: 200px;  /* Added width */
        height: 200px; 
    }
</style>
"""
def display_product_results(images, is_multi_column=False):
    if is_multi_column and len(images) > 1:
        cols = st.columns(min(len(images), 5))  # Limit to 5 columns max
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="bordered-image"><img src="{img["url"]}" width="300"></div>',
                            unsafe_allow_html=True)
                # st.image(img["url"], caption=img["caption"], width=150)
                # st.write(f"**{img['caption']}**")
                st.write(f"Brand: {img['brand']}")
                st.write(f"Price: ${img['price']}")

                st.write(img['description'])
                # st.write(f"Availability: {img['availability']}")
                if img["availability"] == "In Stock":
                    st.button("Buy Now", key=f"buy_{img['id']}")
                else:
                    st.write("Not In Stock")
    else:
        for img in images:
            st.markdown(f'<div class="bordered-image"><img src="{img["url"]}" width="300"></div>',
                        unsafe_allow_html=True)
            # st.image(img["url"], caption=img["caption"], width=200)
            # st.write(f"**{img['caption']}**")
            st.write(f"Brand: {img['brand']}")
            st.write(f"Price: ${img['price']}")

            st.write(img['description'])
            # st.write(f"Availability: {img['availability']}")
            if img["availability"] == "In Stock":
                st.button("Buy Now", key=f"buy_{img['id']}")
            else:
                st.write("Not In Stock")


def generate_personal_intro(query: str) -> str:
    prompt = f"""
    The user has said: "{query}"
    Generate a brief, friendly introduction that acknowledges any personal context or event mentioned.
    If no personal context is given, provide a warm, engaging greeting and ask user to try out some baking products from the redman shop.
    Keep the response concise, within 1-2 sentences. Miantain the flow as after your response query will be done to neo4j and products will be displayed as you are a Hybrid chatbot
    """
    response = llm.invoke(prompt)
    return response.content


import json
from typing import List, Dict

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_llm_response(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse the JSON response from the language model, with error handling and fallback mechanisms.

    Args:
    response_content (str): The raw response content from the language model.

    Returns:
    Optional[Dict[str, Any]]: Parsed JSON response or None if parsing fails.
    """
    try:
        # First, try to parse the entire response as JSON
        return json.loads(response_content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        try:
            # Look for the start and end of a JSON object
            start = response_content.find('{')
            end = response_content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response_content[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # If JSON extraction fails, try to create a structured response
        try:
            lines = response_content.split('\n')
            structured_response = {}
            current_key = None
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    structured_response[key.strip()] = value.strip()
                    current_key = key.strip()
                elif current_key:
                    structured_response[current_key] += ' ' + line.strip()
            if structured_response:
                return structured_response
        except Exception as e:
            logger.error(f"Failed to create structured response: {e}")

    logger.error(f"Failed to parse LLM response: {response_content}")
    return None


def generate_general_info(query: str) -> dict:
    """
    Generate a response with product suggestions from the Neo4j database.

    Args:
    query (str): The user's query.
    db (Neo4jDatabase): An instance of the Neo4j database connection.

    Returns:
    dict: A response containing text and product information.
    """
    prompt = f"""
    The user has said: "{query}"
    You're a shopping assistant for Redman, an AI assistant for an e-commerce platform.
    Suggest a category of products that might interest the user based on their query.
    Your response should be a single word or short phrase representing a product category.
    """

    try:
        response = llm.invoke(prompt)
        logger.debug(f"LLM category suggestion: {response.content}")
        category = response.content.strip().lower()
    except Exception as e:
        logger.error(f"Error getting category suggestion from LLM: {e}")
        category = "baking"  # Default to baking category if LLM fails

    try:
        products = get_products_by_category(db, category)
        if not products:
            products = json.loads(get_random_products(db))
    except Exception as e:
        logger.error(f"Error fetching products from database: {e}")
        products = []

    if not products:
        return fallback_response()

    return format_response(query, category, products)


def get_products_by_category(db: Neo4jDatabase, category: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Fetch products from Neo4j based on the suggested category."""
    cypher_query = """
    MATCH (p:Product)
    WHERE toLower(p.category) CONTAINS toLower($category) OR toLower(p.name) CONTAINS toLower($category)
    RETURN p.name AS name, p.price AS price, p.description AS description, 
           p.image_link AS image_link, p.availability AS availability, p.brand AS brand
    LIMIT $limit
    """
    results = db.run_query(cypher_query, {"category": category, "limit": limit})
    return [dict(record) for record in results]


def get_random_products(db: Neo4jDatabase, limit: int = 3) -> str:
    """Fetch random products from Neo4j as a fallback."""

    query = ""
    parts = query.split('|')
    search_query = parts[0].strip()
    brand = parts[1].strip() if len(parts) > 1 else None

    cypher_query = """
            MATCH (p:Product)
            WHERE (p.name =~ $name_query OR p.description =~ $name_query)
            """

    params = {
        "name_query": f"(?i).*{search_query}.*"
    }

    if brand:
        cypher_query += """
                AND toLower(p.brand) = toLower($brand)
                """
        params["brand"] = brand

    cypher_query += """
            RETURN p.name, p.price, p.description, p.image_link, p.availability, p.quantity, p.brand
            LIMIT 5
            """

    print(f"CYPHER QUERY: {cypher_query}")
    print(f"PARAMS: {params}")

    results = db.run_query(cypher_query, params)
    print(f"QUERY RESULTS: {results}")
    return json.dumps([dict(r) for r in results])


def format_response(query: str, category: str, products: List[Dict[str, Any]]) -> dict:
    """Format the final response with text and product information."""
    response_text = (f"Based on your interest in '{query}', I think you might like some of our {category} products. "
                     f"Here are a few suggestions:")

    images = [
        {
            "url": product.get("image_link", "https://www.redmanshop.com/path/to/default/image.jpg"),
            "caption": product.get("name", "Product Name Not Available"),
            "price": product.get("price", "Price Not Available"),
            "description": product.get("description", "Description Not Available")[:100] + "...",
            "availability": product.get("availability", "Availability Unknown"),
            "brand": product.get("brand", "Redman"),
            "id": f"{product.get('name', 'unknown')}_{time.time()}"
        }
        for product in products
    ]

    return {
        "text": response_text,
        "images": images,
        "timestamp": time.time(),
        "last_product_context": category
    }


def fallback_response() -> dict:
    """Provide a fallback response when no products are found."""
    return {
        "text": "I apologize, but I couldn't find specific products matching your request. "
                "Here's a general recommendation from our popular items:",
        "images": [{
            "url": "https://www.redmanshop.com/path/to/default/image.jpg",
            "caption": "Redman's Best Seller Cake Mix",
            "price": "$9.99",
            "description": "A versatile cake mix perfect for various soft desserts.",
            "availability": "In Stock",
            "brand": "Redman",
            "id": f"default_product_{time.time()}"
        }],
        "timestamp": time.time(),
        "last_product_context": None
    }


def extract_json_from_string(s):
    """Extract the first JSON object from a string."""
    start = s.find('{')
    if start == -1:
        return None

    bracket_count = 0
    for i, char in enumerate(s[start:], start):
        if char == '{':
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            if bracket_count == 0:
                return s[start:i + 1]

    return None

def encode_image(image_url):
    return base64.b64encode(image_url.encode()).decode()

def process_query(query: str,last_product_context: str = None) -> str:
    logger.debug(f"Processing query: {query}")

    # Classification step
    classification_task = Task(
        description=f"""Classify the following query and extract relevant information: '{query}'
        Return your response as a JSON object with the following structure:
        {{
            "query_type": "Recipe Query" or "Product Search" or "Brand Specification" or "Unknown",
            "recipe_name": "extracted recipe name" (only for Recipe Queries),
            "product_name": "extracted product name" (only for Product Searches),
            "brand": "extracted brand" (only for Brand Specification),
            "additional_info": "any other relevant information"
        }}""",
        agent=query_classifier
    )
    classification_crew = Crew(
        agents=[query_classifier],
        tasks=[classification_task],
        verbose=2
    )
    classification_result = classification_crew.kickoff()
    logger.debug(f"Classification result (raw): {classification_result}")

    try:
        # Extract JSON from the classification result
        json_str = extract_json_from_string(classification_result)
        if json_str is None:
            raise ValueError("No JSON object found in the classification result")

        # Parse the extracted JSON
        parsed_result = json.loads(json_str)
        logger.debug(f"Parsed classification result: {parsed_result}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse classification result: {e}")
        return "I'm sorry, but I encountered an error processing your query. Can you please try again?"

    query_type = parsed_result.get("query_type", "Unknown")

    personal_intro = generate_personal_intro(query)
    print (f" PERSONAL INTRO: {personal_intro}")

    response = {"text": personal_intro, "images": [], "timestamp": time.time(), "last_product_context": last_product_context}
    if query_type == "Recipe Query":
        logger.debug("Handling Recipe Query")
        recipe_name = parsed_result.get("recipe_name", "")
        if not recipe_name:
            logger.error("No recipe name found in classification result")
            # response["text"] +="I'm sorry, but I couldn't understand the recipe name. Can you please rephrase your query?"
            # gen= generate_general_info(response["text"])
            # print (f"GENRAL INFO - { gen}")
            response["text"] += personal_intro
            results = json.loads(get_random_products(db, 3))
            for i, product in enumerate(results):
                response["images"].append({
                    "url": product['p.image_link'],
                    "caption": product['p.name'],
                    "price": product['p.price'],
                    "brand": product['p.brand'],
                    "description": f"{product['p.description'][:100]}... | Quantity: {product['p.quantity']}",
                    "availability": product['p.availability'],
                    "id": f"{product['p.name']}_{response['timestamp']}_{i}"
                })
            return  response

        recipe_info = json.loads(recipe_check_tool._run(recipe_name))
        logger.debug(f"Recipe search result: {recipe_info}")

        if recipe_info["found"]:
            product = recipe_info["product"]
            matched_recipe = recipe_info["recipe"]
            original_query = recipe_info["original_query"]

            # response["text"] += f"It looks like you're searching for '{matched_recipe}'. "
            response[
                "text"] += f"A {matched_recipe} recipe uses {product} as a main ingredient. Here are the details of {product}:\n\n"

            # if original_query.lower() != matched_recipe.lower():
            #     response += f"I've corrected the spelling from '{original_query}' to '{matched_recipe}'. "

            # response += f"This recipe uses {product} as a main ingredient. Here are the details of {product}:\n\n"

            product_details_json = product_details_tool._run(product)
            product_details = json.loads(product_details_json)
            print (f"PRODUCT DETAILS :::{product_details}")
            if product_details:
                # response["text"] += f"Name: {product_details['p.name']}\n"
                # response["text"] += f"Price: ${product_details['p.price']}\n"
                # response["text"] += f"Description: {product_details['p.description']}\n"
                # response["text"] += f"Brand: {product_details.get('p.brand', 'N/A')}\n"
                if product_details['p.image_link']:
                    response["images"].append({
                        "url": product_details['p.image_link'],
                        "caption": product_details['p.name'],
                        "price": product_details['p.price'],
                        "description": product_details['p.description'][:100] + "...",  # Truncate description
                        "availability": product_details['p.availability'],
                        "brand": product_details.get('p.brand', 'N/A'),
                        "id": f"{product_details['p.name']}_{response['timestamp']}"
                    })
                else:
                    response["text"] += "\n(No image available)\n"
                availability = product_details['p.availability']
                # if availability == "In Stock":
                #         response["text"] += "\n[Buy Now]\n"
                # else:
                #         response["text"] += "\nNot In Stock\n"
            else:
                response["text"] += "I'm sorry, but I couldn't find detailed information for this product."
        else:
            response["text"]  = f"I'm sorry, but I couldn't find a recipe for '{recipe_info['original_query']}' in our database. Can you please check the spelling or try a different recipe?"
            # gen = generate_general_info(response["text"])
            # print(f"GENRAL INFO - {gen}")
            response["text"] += personal_intro
            results = json.loads(get_random_products(db, 3))
            for i, product in enumerate(results):
                response["images"].append({
                    "url": product['p.image_link'],
                    "caption": product['p.name'],
                    "price": product['p.price'],
                    "brand": product['p.brand'],
                    "description": f"{product['p.description'][:100]}... | Quantity: {product['p.quantity']}",
                    "availability": product['p.availability'],
                    "id": f"{product['p.name']}_{response['timestamp']}_{i}"
                })

    elif query_type == "Brand Specification" and last_product_context:
        brand = parsed_result.get("brand", "")
        print (f"brand:: {brand}")
        search_input = f"{last_product_context} | {brand}"
        print(f"SEARCH INPUT :: {search_input}")
        results = json.loads(product_search_tool._run(search_input))

        if results:
            response[
                "text"] += f"I found {len(results)} product(s) matching '{last_product_context}' with the specified brand '{brand}'. Here they are:\n\n"
            for i, product in enumerate(results):
                response["images"].append({
                    "url": product['p.image_link'],
                    "caption": product['p.name'],
                    "price": product['p.price'],
                    "brand":product['p.brand'],
                    "description": f"{product['p.description'][:100]}",
                    "availability": product['p.availability'],
                    "id": f"{product['p.name']}_{response['timestamp']}_{i}"
                })
        else:
            response[
                "text"] += f"I'm sorry, but I couldn't find any products matching '{last_product_context}' with the specified quantity '{brand}'. Would you like to see products with different Brands?"

    elif query_type == "Product Search":
        product_name = parsed_result.get("product_name", query)
        print(f":PRODUCT NAME IS  {product_name}")
        results = json.loads(product_search_tool._run(product_name))


        if results:
            print (f"FINAL QUERY IS  :: {query} ::")
            if product_name == "":
                response["text"] += f"Meanwhile you can have a look at following products. Here they are:\n\n"
            else:
                response["text"] += f"I found {len(results)} product(s) matching '{product_name}'. Here they are:\n\n"
            for i, product in enumerate(results):
                response["images"].append({
                    "url": product['p.image_link'],
                    "caption": product['p.name'],
                    "price": product['p.price'],
                    "brand": product['p.brand'],
                    "description": f"{product['p.description'][:100]}",
                    "availability": product['p.availability'],
                    "id": f"{product['p.name']}_{response['timestamp']}_{i}"
                })

            if len(results) > 1:
                response[
                    "text"] += "\nWhich product would you like to know more about? You can also specify a brand name if you're interested in a particular brand."
                response["last_product_context"] = product_name
            else:
                response["text"] += "\nWould you like to add this item to your cart? Or specify a different brand?"
                response["last_product_context"] = results[0]['p.name']
        else:
            response[
                "text"] += f"I'm sorry, but I couldn't find any products matching '{product_name}'. Would you like to try a different search term? Meanwhile you can have a look at following products. Here they are:\n\n"
            # gen = generate_general_info(response["text"])
            # print(f"GENRAL INFO - {response}")
            # response["text"] = personal_intro + " " + gen["text"]
            results = json.loads(get_random_products(db,3))
            for i, product in enumerate(results):
                response["images"].append({
                    "url": product['p.image_link'],
                    "caption": product['p.name'],
                    "price": product['p.price'],
                    "brand": product['p.brand'],
                    "description": f"{product['p.description'][:100]}... | Quantity: {product['p.quantity']}",
                    "availability": product['p.availability'],
                    "id": f"{product['p.name']}_{response['timestamp']}_{i}"
                })


    else:
        logger.debug(f"Unrecognized query type: {query_type}")
        response[
            "text"] += f"Sorry we couldn't find what you are lookig for. Do you have a recipe in mind or maybe some baking items. Just let me know. Meanwhile you can have a look at following products. Here they are:\n\n"
        # gen = generate_general_info(response["text"])
        # print(f"GENRAL INFO - {gen}")
        # response["text"] += personal_intro
        results = json.loads(get_random_products(db, 3))
        for i, product in enumerate(results):
            response["images"].append({
                "url": product['p.image_link'],
                "caption": product['p.name'],
                "price": product['p.price'],
                "brand": product['p.brand'],
                "description": f"{product['p.description'][:100]}... | Quantity: {product['p.quantity']}",
                "availability": product['p.availability'],
                "id": f"{product['p.name']}_{response['timestamp']}_{i}"
            })

    logger.debug(f"Final response: {response}")
    return response
st.set_page_config(page_title="E-commerce Chatbot", page_icon="🛒", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit UI


st.title("🛒 RedmanBakeBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# if "waiting_for_quantity" not in st.session_state:
#     st.session_state.waiting_for_quantity = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_product_context" not in st.session_state:
    st.session_state.last_product_context = None
if "last_product_context" not in st.session_state:
    st.session_state.last_product_context = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            display_product_results(message["images"], message.get("multi_column", False))

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Input validation
    if len(prompt.strip()) == 0:
        st.error("Please enter a valid query.")
    elif len(prompt) > 200:
        st.error("Your query is too long. Please limit it to 200 characters.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Getting the best of Redman for you..."):
                try:
                    response = process_query(prompt, st.session_state.last_product_context)

                    st.markdown(response["text"])

                    is_multi_column = len(response["images"]) > 1
                    display_product_results(response["images"], is_multi_column)

                    # Update last_product_context
                    st.session_state.last_product_context = response.get("last_product_context")

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["text"],
                        "images": response["images"],
                        "multi_column": is_multi_column
                    })
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I'm sorry, but I encountered an error while processing your request. Please try again."
                    })

                # Cleanup
            if st.button("End Chat"):
                st.session_state.messages = []
                st.session_state.last_product_context = None
                db.close()
                st.experimental_rerun()
if __name__ == "__main__":
    # This is used when running the file directly
    pass