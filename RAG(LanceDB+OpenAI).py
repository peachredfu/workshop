import streamlit as st
import os

from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatVertexAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
openai.api_key = st.secrets["openai_key"]

palmembeddings = GooglePalmEmbeddings(google_api_key=st.secrets["palm_api_key"])
openaiembedding = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "UPLOADED")

def prompt_inputs_form():  
	# with st.expander("Reveal Code"): st.code(mc.code_ex7, language='python')
	with st.form("Prompt Template"):
		my_prompt_template = st.text_area("Enter a system prompt template. E.g. You are public officer of Singapore.")
		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
		# return a dictionary of the values
		if submitted:
			st.session_state.prompt_template = my_prompt_template
			st.success(f"Your Promp Template is set to:    \n\n   **{my_prompt_template}**")
			return st.session_state.prompt_template

def display_uploaded_files():
	filelist=[]
	for root, dirs, files in os.walk(UPLOAD_DIRECTORY):
		for file in files:
				#filename=os.path.join(root, file)
				filelist.append(file)
	st.write(f"You have the following files uploaded under **{UPLOAD_DIRECTORY}**")
	st.write(filelist)
 
def document_loader():
	loader = DirectoryLoader(f"{UPLOAD_DIRECTORY}", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
	documents = loader.load()
	#loader = PPyPDFLoader(f"{UPLOAD_DIRECTORY}"/*.pdf")
	#documents = loader.load_and_split()

	# chunk size refers to max no. of chars, not tokens
	text_splitter = RecursiveCharacterTextSplitter(
		separators=['\n\n'],
		chunk_size=300, 
		chunk_overlap=0
	)
	documents = text_splitter.split_documents(documents)
	st.write("No. of Chunks: ", f"**{len(documents)}**")
	st.write("chunk(s): ", documents)
	return documents


def lance_vectorstore_creator():
	loader = DirectoryLoader(f"{UPLOAD_DIRECTORY}", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
	#loader = TextLoader(f"{UPLOAD_DIRECTORY}/abc.txt")
	documents = loader.load()
	# chunk size refers to max no. of chars, not tokens
	text_splitter = RecursiveCharacterTextSplitter(
		separators = ["\n\n"],
		chunk_size=300, 
		chunk_overlap=0
	)
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
				"vector": openaiembedding.embed_query("Hello World"),
				"text": "Hello World",
				"id": "1",
			}
		],
		mode="overwrite",
	)
	db = LanceDB.from_documents(documents, embedding=openaiembedding, connection=table)
	LanceDB_TableUpdateDate(TBL_DIRECTORY)
	st.success(f"LanceDB Table last update @ {st.session_state.TBLdate}")
	st.session_state.lance_vs = db
	return db

from datetime import datetime
def LanceDB_TableUpdateDate(TBL_DIRECTORY):
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
	st.subheader('LanceDB for VectorStore and Embedding by OpenAI', divider='rainbow')
	# initialize vectorstore in session_state
	if "lance_vs" not in st.session_state:
		st.session_state.lance_vs = False

	placeholder = st.empty()
	with placeholder.container():
		display_uploaded_files()

	# Add a button to create vectorstore
	lance_vs_btn = st.button('Create/Update VectorStore!')
	if lance_vs_btn:
		lance_vectorstore_creator()
  
	if st.session_state.lance_vs:
		lance_query = st.text_input("Enter a query")
		if lance_query:
			# k refers to top k relevant chunk(s) return
			docs = st.session_state.lance_vs.similarity_search(query=lance_query, k=2, embedding = palmembeddings)
			for doc in docs:
   				st.markdown(doc.page_content)

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

# llm = ChatVertexAI(
# 		model_name="chat-bison",
# 		max_output_tokens=500,
# 		temperature=0,
# 		top_p=0.8,
# 		top_k=40,
# 		verbose=True,
# 		credentials = google_api_cred,
# 		project=google_api_cred.project_id,
# 	)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def openai_completion_stream(prompt):
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			# {"role": "system", "content": st.session_state.prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

def RAG_withOpenAI():
	if st.session_state.lance_vs:
		vectorstore = st.session_state.lance_vs
	else: 
		vectorstore = lance_vectorstore_creator()
		print("I was called!")
	# retriever = vectorstore.as_retriever()
	
	template_raw = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Answer:
"""

	if '{context}' in st.session_state.prompt_template and '{question}' in st.session_state.prompt_template:
		template = st.session_state.prompt_template
	else:
		template = template_raw
  	
	st.markdown(f"**Your Prompt Template:**    \n\n   {template}")

	input_prompt = PromptTemplate(
		input_variables=["context", "question"],
		template=template,
	)

	# step 1 save the memory from your chatbot
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write("**Memory Data**: ", memory_data)

	# # Showing Chat history
	# for message in st.session_state.msg:
	# 	st.chat_message(message["role"]).markdown(message["content"])
  
	try:
		if prompt := st.text_input("**Question Answering**"):
			# set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			docs = vectorstore.similarity_search(query=prompt, k=2)
			context = " \n".join([doc.page_content for doc in docs])

			input_prompt_formated = input_prompt.format(context=context, question=prompt)
			st.session_state.msg.append({"role": "system", "content": input_prompt_formated})
			with st.chat_message("system", avatar = "🦜"):
				st.markdown(input_prompt_formated)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# streaming function
				for response in openai_completion_stream(input_prompt_formated):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context(
				{"input": prompt}, {"output": full_response}
			)
	except Exception as e:
		st.error(e)


def VertexAI_RAG():
	tab1, tab2, tab3, tab4 = st.tabs(["Upload Docs", "Spliting Chunks", "Vector Creation", "Questioin & Answering"])
	with tab1:
		st.subheader("Upload Documents", divider="rainbow")
	
	with tab2:
		st.subheader("Spliting Chunks Demonstration", divider="rainbow")
		document_loader()
	
	with tab3:
		ex12()
	
	with tab4:
		st.subheader("Question Answering with VectorDB by PALM")
		prompt_inputs_form()
		RAG_withOpenAI()
	
		

def main():
	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "You are a helpful assistant"
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=1)
	if "msg" not in st.session_state:
		st.session_state.msg = []
  
	VertexAI_RAG()
		
if __name__ == "__main__":
	main()