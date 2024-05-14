from vanna.ollama import Ollama
from vanna.vannadb import VannaDB_VectorStore
from vanna.chromadb import ChromaDB_VectorStore
import sys
import pandas as pd
import os 
import json

with open("config.json", mode="r") as json_file:
    config_data = json.load(json_file) 


import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y%m%d %H%M%S')
                    
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={  'model': 'llama3' ,'path': 'chromadb'})

# There's usually a library for connecting to your type of database. Any SQL database will work here -- you just have to use the right library.

print( f"connecting ... ")
# fill this with your connection object     

# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()
print(training_data)
print(f"\nTraining Data Size:\n {training_data.size}")
 
if training_data.size>0:
    for id in training_data['id']:
        vn.remove_training_data(id)
else: 

    # Folder path
    gen_folder_path = 'gendata'


    # Load DDLs ---------------------------------------------------------

    # List all files and directories
    ddls_folder_path = f"{gen_folder_path}\\ddls" 

    # Iterate over each file
    for file_name in os.listdir(ddls_folder_path) :
        file_path = os.path.join(ddls_folder_path, file_name)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            # Open the file and read its content
            with open(file_path, 'r') as file:
                file_content = file.read() 
                
                # Train DDL
                vn.train(ddl=file_content)

    # Load DDLs ---------------------------------------------------------


    # Load REFMs ---------------------------------------------------------

    # List all files and directories
    refmss_folder_path = f"{gen_folder_path}\\refms" 

    # Iterate over each file
    for file_name in os.listdir(refmss_folder_path) :
        file_path = os.path.join(refmss_folder_path, file_name)
        
        # Check if it's a file
        if os.path.isfile(file_path):
            # Open the file and read its content
            with open(file_path, 'r') as file:
                file_content = file.read() 
                
                # Train Documentation
                vn.train(documentation=file_content)

    # Load REFMs ---------------------------------------------------------

    # Load SQLs ---------------------------------------------------------
    # Read the Excel file into a DataFrame
    excel_file = 'sql_training_data.xlsx'
    df = pd.read_excel(excel_file)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        description = row['DESCRIPTION']
        query = row['QUERY']
         
        # Train SQL
        vn.train(
            question=description, 
            sql=query
        )

    # Load SQLs ---------------------------------------------------------

    # Sample question ---------------------------------------------------------
    with open("prompt01.txt", "r") as file_prompt:
        prompt_content = file_prompt.read() 

    # Replace placeholders in the prompt content 
    prompt_content = prompt_content.replace("##ME##", config_data.get("user-me"))
    vn.train(documentation =prompt_content )

    vn.ask(question = prompt_content + "User question: medicaid cases opened in last 5 years", print_results=False, visualize=False, auto_train=True)
    # Sample question ---------------------------------------------------------

    training_data = vn.get_training_data()
    print(f"\nTraining Data added !\nsize: {training_data.size}")


   

 
