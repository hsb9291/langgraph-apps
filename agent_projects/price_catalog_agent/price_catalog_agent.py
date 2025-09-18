import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utilities.common_agent_library import create_agent

# 1. Load environment variables
load_dotenv()

# 2. Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Define the Agent's Specific Data and Tools
# The catalog is passed as a string, as requested.
PRICE_CATALOG_DATA = """
--- Restaurant Menu ---
Beverages:
- Soda: $2.50
- Iced Tea: $3.00
- Coffee: $2.00
- Fresh Orange Juice: $4.50

Appetizers:
- Mozzarella Sticks: $8.99
- Loaded Nachos: $12.50
- Chicken Wings (6 pc): $10.99
- Onion Rings: $7.99

Main Courses:
- Classic Cheeseburger: $15.99
- Margherita Pizza (12-inch): $18.50
- Grilled Salmon with Asparagus: $22.75
- Steak Frites: $25.50

Desserts:
- New York Cheesecake: $7.50
- Chocolate Lava Cake: $8.00
- Ice Cream Sundae: $6.50
"""
tools = [] # This agent doesn't need any special tools

# 4. Define the Agent's Specific System Message
system_message_content = f"""
You are a restaurant price catalog agent. Your task is to answer user questions
about menu items and their prices using the provided catalog data.

Price Catalog:
{PRICE_CATALOG_DATA}

Respond concisely and in a helpful manner. Do not mention items that are not in the catalog.
"""

# 5. Create the Agent using the common library
price_catalog_agent = create_agent(
    llm=llm,
    system_message_content=system_message_content,
    tools=tools
)

'''
# Example Usage
if __name__ == "__main__":
    print("Price Catalog Agent created!")
    for s in price_catalog_agent.stream({"messages": [("user", "How much is the Classic Cheeseburger?")]}):
        print(s)
    print("\n---")
    for s in price_catalog_agent.stream({"messages": [("user", "What kind of desserts do you have?")]}):
        print(s)
    print("\n---")
    for s in price_catalog_agent.stream({"messages": [("user", "Do you have French onion soup?")]}):
        print(s)
'''