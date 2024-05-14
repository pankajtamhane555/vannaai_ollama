from vanna.ollama import Ollama
from vanna.vannadb import VannaDB_VectorStore
from vanna.chromadb import ChromaDB_VectorStore
import sys
import pandas as pd
import os 
import json

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y%m%d %H%M%S')
                    
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={  'model': 'llama3' ,'path': 'chromadb'})
 
 
#prompt_content = "Scenario: You are an AI assistant specializing in data analysis and visualization, with expertise in DB2 SQL server syntax. A human will provide a data schema which you will use to generate SQL queries.\nResponse Format: string (\"DB2 SQL query\")\n\nUser question: "
#result = vn.ask(question = prompt_content + "medicaid cases opened and assigned to me in last 5 years", print_results=True, visualize=False, auto_train=True)
#result = vn.ask(question = sys.argv[1], print_results=False, visualize=False)
#result =vn.ask_question("cases that are opened in last 6 months")

#print(result)

#gsql = vn.generate_sql("tanf cases that are assigned to central office")
#gsql = vn.generate_sql("medicaid cases opened in last 5 years")
#gsql = vn.generate_sql("persons with bad address")
#gsql = vn.generate_sql("cases that are assigned to Senthil")
gsql = vn.generate_sql("show me NCPs with bad address")
print(gsql) 

