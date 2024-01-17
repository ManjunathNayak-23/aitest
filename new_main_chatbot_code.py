# Step1
# import the required libraries
import openai
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents.types import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import json
from langchain.chains import RetrievalQAWithSourcesChain


class BotAssistant():
    
    def __init__(self):
        self.pinecone_api_key = "91c8323c-8073-4d97-8e58-27047cc36d6f"
        self.pinecone_environment="gcp-starter"
        self.index_name = "packages"

    def pinecone_conn(self):
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
        return pinecone
        
    def get_embeddings(self):
        embeddings = AzureOpenAIEmbeddings(
            api_key="7b3b1ef5555e4d16a5294c6517abc18d",
            azure_endpoint="https://alg-ai-ml-openai-dev-03.openai.azure.com/",
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
            openai_api_type= "azure"
        )
        return embeddings
    
    def create_llm(self):
        llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo-16k",
            model_name="gpt-35-turbo-16k",
            openai_api_key="7b3b1ef5555e4d16a5294c6517abc18d",
            openai_api_type="Azure",
            azure_endpoint="https://alg-ai-ml-openai-dev-03.openai.azure.com/",
            openai_api_version="2023-05-15"
        )
        return llm
    
    def get_system_message(self):
        system_message = """
            "You are an AI assistant and your only goal is to help users to find the vacation packages.
             Provide the travel package answers based on the vector data or retrieval data."
            "Do not answer questions that are not related to vacation packages."
            "If the requested information is not available in the retrieved data you can say 'Can you be more specific'."
            
        """
        return system_message
    
    def get_tool(self, chain):
        tools = [
            Tool(
                name="wiki-tool",
                func=chain.run,
                description="Useful when you need to answer travel package questions",
            )
        ]
        return tools
    
    def chat_assistant(self, query):
        pinecone = self.pinecone_conn()
        index = pinecone.Index(self.index_name)
        # text_field = "HotelName"
        text_field = "HotelName, HotelTheme, HotelAddress, HotelRating, OriginName, DestinationName, PackagePrice"
        embeddings = self.get_embeddings()
        llm = self.create_llm()
        # vectorstore = Pinecone(
        #     index, embeddings, text_field
        # )
        vectorstore = Pinecone(index, embeddings, text_field)
        similar_vector_data = vectorstore.similarity_search(query, k=3)
        vectorstore.similarity_search(query, k=3)
        system_message = self.get_system_message()
        conversational_memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=5,
                return_messages=True
            )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )
        tools = self.get_tool(chain)

        executor = initialize_agent(
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            memory=conversational_memory,
            agent_kwargs={"system_message": system_message},
            verbose=True
        )
        final_response = executor.invoke(input=query)
        json_response = json.dumps(final_response)
        # print('Results: \n', json_response) 
        return json_response
        

    

"""
# Step2
# connect to pinecone client
pinecone_api_key = "91c8323c-8073-4d97-8e58-27047cc36d6f"
pinecone_environment="gcp-starter"
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = "packages"

# query = "show me list of all the hotels from the destination Jamaica, Montego Bay?"
# query = "who are you?"
# query = "Help me plan a vacation"
# query = "I am planning to travel in month of April and my budget is below 2000, can you show me best travel packages?"
# query = "is there any destination name with this Jamaica, Montego Bay?"
# query = "how many counties in New York?"
# query = "which is the best hotel name in the destination Jamaica?"
query = "can you provide me the hotelthemes for the hotel Riu Palace in the destination Jamaica Montego Bay?"
# query = "show me a travel package to Jamaica?"

# Step3
# Create azureopenai embeddings
embeddings = AzureOpenAIEmbeddings(
            api_key="7b3b1ef5555e4d16a5294c6517abc18d",
            azure_endpoint="https://alg-ai-ml-openai-dev-03.openai.azure.com/",
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
            openai_api_type= "azure"
        )

# Step4
# create an index through langchain vectorstore Pinecone module
index = pinecone.Index(index_name)


text_field = "HotelName"
vectorstore = Pinecone(
    index, embeddings, text_field
)

vectorstore.similarity_search(query, k=3)

# doc_search = Pinecone.from_existing_index(index_name, embeddings)
# doc_search.similarity_search(query)

llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo-16k",
            model_name="gpt-35-turbo-16k",
            openai_api_key="7b3b1ef5555e4d16a5294c6517abc18d",
            openai_api_type="Azure",
            azure_endpoint="https://alg-ai-ml-openai-dev-03.openai.azure.com/",
            openai_api_version="2023-05-15"
        )

"""
system_message = """
            "You are an AI assistant and your only goal is to help users to find the vacation packages."
            "Do not answer questions that are not related to vacation packages."
            "If the requested information is not available in the retrieved data you can say 'Can you be more specific'."
            
        """

# template = PromptTemplate(input_variables=["question"], template=system_message)
"""
conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )

chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )

# qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )

tools = [
        Tool(
            name="wiki-tool",
            func=chain.run,
            description="Useful when you need to answer travel package questions",
        )
        ]

executor = initialize_agent(
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            memory=conversational_memory,
            agent_kwargs={"system_message": system_message},
            verbose=True
        )
        
# final_response = executor.run(query)['answer']
final_response = executor.invoke(input=query)
# json_response = json.dumps(final_response)
# print(json_response)



# final_response=qa_with_sources(query)
json_response = json.dumps(final_response)
print('Results: \n', json_response)
"""
