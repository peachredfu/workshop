### pip install google-auth google-cloud-aiplatform tiktoken pinecone-client
import streamlit as st
import os

# Exercise 1: title, text_input, function
def ex1():
	st.title("My First Application")
	query = st.text_input("say something")
	if query:
		st.write("You entered:  \n   " + query)


# Exercise 2: radio button
def ex2():
	st.title("Streamlit Radio Button")
	llm = st.radio(
		"Your preferred LLM",
		[":rainbow[GPT3.5]", "**PALM**"],
		captions = ["OpenAI", "Google"], horizontal=True)

	if llm == ':rainbow[GPT3.5]':
		st.write('You selected GPT3.5')
	elif llm == '**PALM**':
		st.write('You selected PALM')
	else:
		st.write("You didn\'t select LLM.")

	
# Exercise 3: sidebar
def ex3():
	clear_session = st.sidebar.button("**Clear Session**")
	if clear_session: 
		st.session_state.clear()
		print("you session is cleared")
	
	functions = [
	   			"Function 1", 
				"Function 2",
	   			"Function 3", 
				"Function 4",
	   			"Function 5", 
				"Function 6",
	   			"Function 7", 
				"Function 8",
	   			"Function 9", 
				"Function 10",
				"Function 11",
				"Function 12",
				"Function 13",
				"Function 14",
				"Function 15",
			]
	# if 'functions' not in st.session_state: 
	# 	st.session_state.fuctnions = []
	# st.session_state.fuctnions = functions
  
	opt = st.sidebar.radio(
	  		label=":rainbow[Select a Function]",
			options=functions,
			captions = [
				"Stremlit Input Output and Variables",	#1
				"Stremlit Radio Button",	#2
				"Stremlit Sidebar",	#3
				"Stremlit Chat Elements",	#4
				"Stremlit Upload Documents",	#5
				"Rule-based Echo Chatbot ",	#6
				"Stremlit Prompt Input Form",	#7
				"Chatbot using OpenAI API",	#8
				"Chatbot using OpenAI API with Stream",	#9
				"Chatbot with Memory",	#10
				"Chatbot using PALM API",	#11   
				"LanceDB for VectorStore",	#12
				"Pinecone for VectorStore",	#13
				"Chatbot using Vertex AI",	#14
				"Chatbot using Vertex AI with Stream"	#15    
			], 
   			horizontal=False)
	
	# if 'opt' not in st.session_state: 
	# 	st.session_state.opt = st.sidebar.radio("",options=functions)
	# st.session_state.opt = opt

	if opt == 'Function 1': ex1()
	elif opt == 'Function 2': ex2()
	elif opt == 'Function 3': st.write("Sidebar on Your Left")
	elif opt == 'Function 4': ex4()
	elif opt == 'Function 5': ex5()
	elif opt == 'Function 6': ex6()
	elif opt == 'Function 7': prompt_inputs_form()
	elif opt == 'Function 8': ex8()
	elif opt == 'Function 9': ex9()
	elif opt == 'Function 10': ex10()
	elif opt == 'Function 11': ex11()
	elif opt == 'Function 12': ex12()
	elif opt == 'Function 13': ex13()
	elif opt == 'Function 14': ex14()
	elif opt == 'Function 14': ex15()
	else: ex1()
 

# Exercise 4: chat elements
def ex4():
	st.title("Chat Elements")
	msg_container = st.container()
	with msg_container:
		user_message = st.chat_message("user")
		user_message.write("I'm user")

		asst_message = st.chat_message("assistant")
		asst_message.write("I'm your assistant")

		other_message = st.chat_message("whoami")
		other_message.write("I'm nobody")

		st.chat_message("user", avatar="👩‍🎤").write("I'm user")
		st.chat_message('bot', avatar="🤖").write("I'm bot!")

		st.chat_message('PA assistant', avatar='./avatar.png').write("Hi, I'm PA assistant")
  
		st.chat_message('assistant', avatar='https://raw.githubusercontent.com/dataprofessor/streamlit-chat-avatar/master/bot-icon.png').write('Hello world!')
  
		prompt = st.chat_input("Say something")
		if prompt:
			st.chat_message("user").write(f"**You entered**:  {prompt}")


# Exercise 5: upload files to working directory
# Create a folder for uploaded files, create only when it doesn't exist
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "UPLOADED")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
# Define allowed file extensions and maximum file size
allowed_extensions = [".docx", ".txt", ".pdf"]
max_file_size = 10 * 1024 * 1024  # 10MB

def display_uploaded_files():
	filelist=[]
	for root, dirs, files in os.walk(UPLOAD_DIRECTORY):
		for file in files:
				#filename=os.path.join(root, file)
				filelist.append(file)
	st.write(f"You have the following files uploaded under **{UPLOAD_DIRECTORY}**")
	st.write(filelist)

# Function to check if the file is valid
def is_valid_file(file):
	if file is None:
		return False
	file_extension = os.path.splitext(file.name)[-1]
	file_size = len(file.getvalue())
	return file_extension in allowed_extensions and file_size <= max_file_size

# uploading component
def ex5():
	# File upload section
	uploaded_file = st.file_uploader("Upload a file", 
									 type=allowed_extensions)

	if uploaded_file is not None:
		if is_valid_file(uploaded_file):
			# Save the uploaded file to the upload folder
			file_path = os.path.join(f"{UPLOAD_DIRECTORY}", uploaded_file.name)
			with open(file_path, "wb") as f:
				f.write(uploaded_file.getvalue())
			st.success(f"File '{uploaded_file.name}' uploaded successfully.")
		else:
			st.error("Invalid file. Please upload a .docx, .txt, or .pdf file with a size of up to 10MB.")
	display_uploaded_files()


#Exercise 6 : Session State, Rule-based Echo Chatbot 
def ex6():
	st.title("Rule-based Echo Bot")
	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		st.chat_message(message["role"]).markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What is up?"):
		# Display user message in chat message container
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.messages.append({"role": "user", "content": prompt})

		# define rule-based response
		response = ""
		if prompt.lower() == "hello":
			response = "Hi there what can I do for you"
		else:
			response = f"Echo: {prompt}"
   
		# Display assistant response in chat message container
		st.chat_message("assistant").markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})
  

# Exercise 7 : prompt input form
def prompt_inputs_form():  # Using st.form, create the starting prompt to your prompt template, this is an expert on a topic that is talking to a user of a certain age
	# langchain prompt template
	with st.form("Prompt Template"):
		my_prompt_template = st.text_input("Enter a system prompt template. E.g. You are public officer of Singapore.")
		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
		# return a dictionary of the values
		if submitted:
			st.session_state.prompt_template = my_prompt_template
			print(f"you session_state for prompt_template is set to <{my_prompt_template}>")
			return st.session_state.prompt_template


# Exercise 8 : Using the OpenAI API
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
openai.api_key = st.secrets["openai_key"]

def openai_completion(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
	)
	return response

# integration API call into chat components
def ex8():
	st.title("Chatbot using OpenAI API")
	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "You are a helpful assistant"
  	
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Display chat messages from history on app rerun
	for message in st.session_state.msg:
		st.chat_message(message["role"]).markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What is up?"):
		# Display user message in chat message container
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.msg.append({"role": "user", "content": prompt})

		# call OpenAI API to get response
		response_raw = openai_completion(prompt)
		response = response_raw["choices"][0]["message"]["content"].strip()
		total_tokens = str(response_raw["usage"]["total_tokens"])
  
		# Display assistant response in chat message
		st.chat_message("assistant").markdown(response)
		c = st.empty()
		c.markdown(f"**Total tokens used in last converstation:** {total_tokens}")
  
		# Add assistant response to chat history
		st.session_state.msg.append({"role": "assistant", "content": response})


# Exercise 9 : Using the OpenAI API with streaming option
def openai_completion_stream(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

# integration API call into streamlit chat components
def ex9():
	st.title("Chatbot using OpenAI API with Streaming")
	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "You are a helpful assistant"
  
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	if prompt := st.chat_input("say something"):
		# set user prompt in chat history
		st.session_state.msg.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)

		with st.chat_message("assistant"):
			message_placeholder = st.empty()
			full_response = ""
			# streaming function
			for response in openai_completion_stream(prompt):
				full_response += response.choices[0].delta.get("content", "")
				message_placeholder.markdown(full_response + "▌")
			message_placeholder.markdown(full_response)
		st.session_state.msg.append({"role": "assistant", "content": full_response})



# Exercise 10: Chatbot with memory
import pandas as pd
from langchain.memory import ConversationBufferWindowMemory
def ex10():
	st.title("Chatbot with Memory")
	# call prompt_inputs_form to get newly submitted prompt template
	input_prompt = prompt_inputs_form()
	if pd.isnull(input_prompt):
		input_prompt = "You are a helpful assistant"

	if "memory" not in st.session_state: # 3 means that the bot remember the last 3-rounds of converstaions
		st.session_state.memory = ConversationBufferWindowMemory(k=3) 

	# step 1 save the memory from your chatbot
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("**Instruction Prompt Setup**: ", input_prompt)
	st.write("**Memory Data**: ", memory_data)
 
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	st.session_state.prompt_template = f"""
{input_prompt}										

Below is the conversation history between the AI and Users so far

{memory_data}

"""
	# call the function in your base bot
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing Chat history
	for message in st.session_state.msg:
		st.chat_message(message["role"]).markdown(message["content"])
  
	try:
		if prompt := st.chat_input("say something?"):
			# set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# streaming function
				for response in openai_completion_stream(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context(
				{"input": prompt}, {"output": full_response}
			)
	except Exception as e:
		st.error(e)


# Exercise 11: Using PALM API (without stream)
import google.generativeai as palm
# set the PALM API key.
os.environ["PALM_API_KEY"] = st.secrets["palm_api_key"]
palm.configure(api_key=st.secrets["palm_api_key"])

# Call the PALM API and print the response.
def palm_chat(prompt):
	response = palm.chat(messages=prompt)
	print(response.last)
	return response.last

def ex11():
	st.title("Chatbot using PALM API")
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("say something"):
			# set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = palm_chat(prompt)
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)
	

# Exercise 12: LanceDB for Text Embeddings and Similarity Search
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
palmembeddings = GooglePalmEmbeddings(google_api_key=st.secrets["palm_api_key"])

def lance_vectorstore_creator():
	loader = TextLoader(f"{UPLOAD_DIRECTORY}/tmp.txt")
	# loader = PyPDFLoader(f"{os.getcwd}/uploaded_files/*.pdf")
	documents = loader.load()
	 # chunk size refers to max no. of chars, not tokens
	text_splitter = CharacterTextSplitter(separator = "\n\n", 
											chunk_size=200, 
											chunk_overlap=0)

	documents = text_splitter.split_documents(documents)
 
	# Create a folder for VectorDB, and create only when it doesn't exist
	DB_DIRECTORY = os.path.join(os.getcwd(), "LanceDB")
	os.makedirs(DB_DIRECTORY, exist_ok=True)
	TABLE_NAME = "my_table"
	TBL_DIRECTORY = os.path.join(DB_DIRECTORY, TABLE_NAME+'.lance')

	db = lancedb.connect(DB_DIRECTORY)
	table = db.create_table(
		TABLE_NAME,
		data=[
			{
				"vector": palmembeddings.embed_query("Hello World"),
				"text": "Hello World",
				"id": "1",
			}
		],
		mode="overwrite",
	)
	db = LanceDB.from_documents(documents, palmembeddings, connection=table)
	get_TableUpdateDate(TBL_DIRECTORY)
	st.success(f"LanceDB Table last update @ {st.session_state.TBLdate}")
	return db

from datetime import datetime
def get_TableUpdateDate(TBL_DIRECTORY):
	if 'TBLdate' not in st.session_state:
		st.session_state.TBLdate = "unknown"
		
	if os.path.exists(TBL_DIRECTORY):
		last_update_timestamp = os.path.getmtime(TBL_DIRECTORY)
		last_update_date = datetime.fromtimestamp(last_update_timestamp).strftime('%Y-%m-%d %H:%M:%S')
		print(f"Last update date of DB  '{TBL_DIRECTORY}': {last_update_date}")
		st.session_state.TBLdate = last_update_date
	else:
		print(f"The folder '{TBL_DIRECTORY}' does not exist or is not a directory.")


def ex12():
	st.subheader('LanceDB for VectorStore (by PAML) and Similarity Search', divider='rainbow')
	
	# initialize vectorstore in session_state
	if "lance_vs" not in st.session_state:
		st.session_state.lance_vs = False

	placeholder = st.empty()
	with placeholder.container():
		display_uploaded_files()

	# Add a button to create vectorstore
	lance_vs_btn = st.button('Create/Update VectorStore!')
	if lance_vs_btn:
		db = lance_vectorstore_creator()
		st.session_state.lance_vs = db
  
	if st.session_state.lance_vs:
		lance_query = st.text_input("Enter a query")
		if lance_query:
			docs = st.session_state.lance_vs.similarity_search(query=lance_query, k=3, embedding = palmembeddings)
			for doc in docs:
   				st.write(doc.page_content)


# Exercise 13: Pinecone for Text Embeddings and Similarity Search
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Pinecone
import pinecone

def pinecone_indexing(index_name):
	loader = TextLoader(f"{UPLOAD_DIRECTORY}/tmp.txt")
	documents = loader.load()
		# chunk size refers to max no. of chars, not tokens
	text_splitter = CharacterTextSplitter(separator = "\n\n", 
											chunk_size=200, 
											chunk_overlap=0)

	documents = text_splitter.split_documents(documents)

	dimensions = len(palmembeddings.embed_query("put anything"))

	# First, check if our index already exists. If it doesn't, we create it
	if index_name not in pinecone.list_indexes():
		# we create a new index
		pinecone.create_index(
				name=index_name,
				metric='cosine',
				dimension=dimensions)
	else:
		pinecone.delete_index(index_name) # delete existing before create new
		pinecone.create_index(
				name=index_name,
				metric='cosine',
				dimension=dimensions)

	# The PALM embedding model uses 768 dimensions; Create Vector DB
	Pinecone.from_documents(documents, palmembeddings, index_name=index_name)
	# connect to index
	index = pinecone.Index(index_name)
	text_field = "text"
	vectorstore = Pinecone(index, palmembeddings, text_field)
	return vectorstore

def ex13():
	st.subheader("Pinecone for VectorStore (by PAML) and Similarity Search", divider='rainbow')
	if "pinecone_vs" not in st.session_state:
		st.session_state.pinecone_vs = False

	placeholder = st.empty()
	with placeholder.container():
		display_uploaded_files()
  
	pinecone.init(api_key=st.secrets["pinecone_key"], environment="asia-northeast1-gcp")
	index_name = "workshop-palm"
 
	# Add a button to create vectorstore
	pinecone_vs_btn = st.button('Create/Update VectorStore!')

	if index_name not in pinecone.list_indexes():
		st.error("pinecone index not exist")
	else:
		index = pinecone.Index(index_name)
		text_field = "text"
		db = Pinecone(index, palmembeddings, text_field)
		st.session_state.pinecone_vs = db
		st.success("pinecone index successfully loaded")

	if pinecone_vs_btn:
		db = pinecone_indexing(index_name)
		st.session_state.pinecone_vs = db
		st.info("pinecone index successfully created/updated")

	if st.session_state.pinecone_vs:
		pinecone_query = st.text_input("Enter a query")
		if pinecone_query:
			docs = st.session_state.pinecone_vs.similarity_search(query=pinecone_query, k=3)
			for doc in docs:
				st.write(doc.page_content)
 

# Exercise 14: Using Vertex AI API (without stream)
from google.oauth2 import service_account
from langchain.chat_models import ChatVertexAI
from langchain.schema import HumanMessage, SystemMessage

google_api_cred = service_account.Credentials.from_service_account_info(
	info={
		"type": st.secrets['type'] ,
		"project_id": st.secrets['project_id'] ,
		"private_key_id": st.secrets['private_key_id'] ,
		"private_key": st.secrets['private_key'] ,
		"client_email": st.secrets['client_email'] ,
		"client_id": st.secrets['client_id'] ,
		"auth_uri": st.secrets['auth_uri'] ,
		"token_uri": st.secrets['token_uri'] ,
		"auth_provider_x509_cert_url": st.secrets['auth_provider_x509_cert_url'] ,
		"client_x509_cert_url": st.secrets['client_x509_cert_url'] ,
		"universe_domain": st.secrets['universe_domain'] 
	},
)

vertex_chat = ChatVertexAI(
		model_name="chat-bison",
		max_output_tokens=500,
		temperature=0,
		top_p=0.8,
		top_k=40,
		verbose=True,
		credentials = google_api_cred,
		project=google_api_cred.project_id,
	)

# integration API call into chat components
def ex14():
	st.title("Chatbot using Vertex AI")
	# Initialize chat history
	# call prompt_inputs_form to get newly submitted prompt template
	input_prompt = prompt_inputs_form()
	if pd.isnull(input_prompt):
		input_prompt = "You are a helpful assistant"

	if "memory" not in st.session_state: # 3 means that the bot remember the last 3-rounds of converstaions
		st.session_state.memory = ConversationBufferWindowMemory(k=1) 

	# step 1 save the memory from your chatbot
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("**Instruction Prompt Setup**: ", input_prompt)
	st.write("**Memory Data**: ", memory_data)
 
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	st.session_state.prompt_template = f"""
{input_prompt}										

Below is the conversation history between the AI and Users so far

{memory_data}

"""

	# call the function in your base bot
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing Chat history
	for message in st.session_state.msg:
		st.chat_message(message["role"]).markdown(message["content"])

	if prompt := st.chat_input("What is up?"):
		# set user prompt in chat history
		st.session_state.msg.append({"role": "user", "content": prompt})
		st.chat_message("user").markdown(prompt)

		with st.chat_message("assistant"):
			message_placeholder = st.empty()
			full_response = vertex_chat([SystemMessage(content=st.session_state.prompt_template) ,HumanMessage(content=prompt)]) # further improve https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm
			message_placeholder.markdown(full_response.content)
		
		st.session_state.msg.append({"role": "assistant", "content": full_response.content})
		st.session_state.memory.save_context(
			{"input": prompt}, {"output": full_response.content}
		)


# Exercise 15: Using VertexAI stream
# integration strem API call into chat components
def ex15():
	st.title("Chatbot using Vertex AI with Streaming")
	# Initialize chat history
	# call prompt_inputs_form to get newly submitted prompt template
	input_prompt = prompt_inputs_form()
	if pd.isnull(input_prompt):
		input_prompt = "You are a helpful assistant"

	if "memory" not in st.session_state: # 3 means that the bot remember the last 3-rounds of converstaions
		st.session_state.memory = ConversationBufferWindowMemory(k=1) 

	# step 1 save the memory from your chatbot
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("**Instruction Prompt Setup**: ", input_prompt)
	st.write("**Memory Data**: ", memory_data)
 
	# step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	st.session_state.prompt_template = f"""
{input_prompt}										

Below is the conversation history between the AI and Users so far

{memory_data}

"""

	# call the function in your base bot
	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Showing chat history
	for message in st.session_state.msg:
		st.chat_message(message["role"]).markdown(message["content"])

	if prompt := st.chat_input("say something"):
		# set user prompt in chat history
		st.session_state.msg.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)
   
		with st.chat_message("assistant"):
			message_placeholder = st.empty()
			full_response = ""
			for response in vertex_chat.stream([SystemMessage(content=st.session_state.prompt_template) ,HumanMessage(content=prompt)]):
				full_response += response.content
				message_placeholder.markdown(full_response + "▌")
			message_placeholder.markdown(full_response)

		st.session_state.msg.append({"role": "assistant", "content": full_response})
		st.session_state.memory.save_context(
			{"input": prompt}, {"output": full_response}
		)


def main():

	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "You are a helpful assistant"
  
	password_container = st.empty()
	password_container.text_input("Enter the password", key="pwd_input", type="password")

	if st.session_state.pwd_input == st.secrets['workshop_pwd']:
		password_container.empty()
		ex3()
	elif st.session_state.pwd_input!='' and st.session_state.pwd_input != st.secrets['workshop_pwd']:
		st.error("Incorrect password. Please enter the correct password to proceed.")



if __name__ == "__main__":
	main()

