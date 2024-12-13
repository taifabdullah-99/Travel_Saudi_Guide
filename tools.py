
# import numpy as np
# import pickle
# from openai import OpenAI
# import faiss

# class SimpleRAG:
#     def __init__(self, max_tokens=1000):
#         self.client = OpenAI()
#         self.index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
#         self.texts = []
#         self.max_tokens = max_tokens

#     def add_documents(self, documents):
#         """Add documents to the vector store"""
#         for doc in documents:
#             embedding = self.client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=doc
#             ).data[0].embedding
#             self.index.add(np.array([embedding]))
#             self.texts.append(doc)
#         pickle.dump(self.index, open('vectors.pkl', 'wb'))
#         pickle.dump(self.texts, open("texts.pkl", "wb"))

#     def load_documents(self):
#         self.index = pickle.load(open('vectors.pkl', 'rb'))
#         self.texts = pickle.load(open('texts.pkl', 'rb'))

#     def retrieve(self, query, k=3):
#         """Retrieve k most relevant documents"""
#         query_embedding = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=query
#         ).data[0].embedding
#         D, I = self.index.search(np.array([query_embedding]), k=k)
#         return [self.texts[i] for i in I[0]]

#     def preprocess_question(self, question):
#         """Preprocess the user question for clarity and specificity"""
#         # Check if the question contains location-related keywords
#         location_keywords = ['where', 'city', 'region', 'place', 'location']
#         if not any(keyword in question.lower() for keyword in location_keywords):
#             question += " Please specify the location or city you're asking about."
#         return question

#     def generate_prompt(self, query, relevant_docs):
#         """Create prompt for the LLM"""
#         context = "\n".join(relevant_docs)
#         prompt = f"""Use the following pieces of context to answer the question.
#         If you cannot find the answer in the context, say "I don’t have enough information to answer this question."
#         If the user asked about specific activities or places, provide the region and city where they can do this activity.
#         Context:
#         {context}
#         Question: {query}
#         Answer:"""
#         return prompt

#     def query(self, question):
#         """Full RAG pipeline"""
#         # Preprocess the user question
#         question = self.preprocess_question(question)
#         # 1. Retrieve relevant documents
#         relevant_docs = self.retrieve(question)
#         # 2. Generate prompt with context
#         prompt = self.generate_prompt(question, relevant_docs)
#         # 3. Get answer from LLM
#         response = self.client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=self.max_tokens
#         )
#         return response.choices[0].message.content


# import numpy as np
# import pickle
# from openai import OpenAI
# import faiss

# class SimpleRAG:
#     def __init__(self, max_tokens=1000):
#         self.client = OpenAI()
#         self.index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
#         self.texts = []
#         self.max_tokens = max_tokens

#     def add_documents(self, documents):
#         """Add documents to the vector store"""
#         for doc in documents:
#             embedding = self.client.embeddings.create(
#                 model="text-embedding-3-small",
#                 input=doc
#             ).data[0].embedding
#             self.index.add(np.array([embedding]))
#             self.texts.append(doc)
#         pickle.dump(self.index, open('vectors.pkl', 'wb'))
#         pickle.dump(self.texts, open("texts.pkl", "wb"))

#     def load_documents(self):
#         self.index = pickle.load(open('vectors.pkl', 'rb'))
#         self.texts = pickle.load(open('texts.pkl', 'rb'))

#     def retrieve(self, query, k=3):
#         """Retrieve k most relevant documents"""
#         query_embedding = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=query
#         ).data[0].embedding
#         D, I = self.index.search(np.array([query_embedding]), k=k)
#         return [self.texts[i] for i in I[0]]

#     def preprocess_question(self, question):
#         """Preprocess the user question for clarity and specificity"""
#         # Check if the question contains location-related keywords
#         location_keywords = ['where', 'city', 'region', 'place', 'location']
#         if not any(keyword in question.lower() for keyword in location_keywords):
#             question += " Please specify the location or city you're asking about."
#         return question

#     def generate_prompt(self, query, relevant_docs):
#         """Create prompt for the LLM"""
#         context = "\n".join(relevant_docs)
#         prompt = f"""Use the following pieces of context to create a structured travel itinerary.
#         If you cannot find the answer in the context, say "I don’t have enough information to answer this question.
#         If the user mentions a specific activity in a specific destination, ensure that the activity remains tied to the mentioned destination."
#         Please provide the itinerary in the following format:
        
#         Itinerary:
#         - Destination: [Destination Name]
#           - Activities: [List of Activities]
#           - Resturant: [List of Resturants]
#           - Additional Notes: [Any other relevant information]
        
#         Context:
#         {context}
#         Question: {query}
#         Answer:"""
#         return prompt

#     def format_response(self, response):
#         """Format the response into a structured itinerary"""
#         # Here you can implement any additional formatting logic if needed
#         return response.strip()

#     def query(self, question):
#         """Full RAG pipeline"""
#         # Preprocess the user question
#         question = self.preprocess_question(question)
#         # 1. Retrieve relevant documents
#         relevant_docs = self.retrieve(question)
#         # 2. Generate prompt with context
#         prompt = self.generate_prompt(question, relevant_docs)
#         # 3. Get answer from LLM
#         response = self.client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that creates structured travel itineraries."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=self.max_tokens
#         )
#         # Format the response
#         structured_itinerary = self.format_response(response.choices[0].message.content)
#         return structured_itinerary


import numpy as np
import pickle
from openai import OpenAI
import faiss

class SimpleRAG:
    def __init__(self, max_tokens=1000):
        self.client = OpenAI()
        self.index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
        self.texts = []
        self.max_tokens = max_tokens

    def add_documents(self, documents):
        """Add documents to the vector store"""
        for doc in documents:
            embedding = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            ).data[0].embedding
            self.index.add(np.array([embedding]))
            self.texts.append(doc)
        pickle.dump(self.index, open('vectors.pkl', 'wb'))
        pickle.dump(self.texts, open("texts.pkl", "wb"))

    def load_documents(self):
        self.index = pickle.load(open('vectors.pkl', 'rb'))
        self.texts = pickle.load(open('texts.pkl', 'rb'))

    def retrieve(self, query, k=3):
        """Retrieve k most relevant documents"""
        query_embedding = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        D, I = self.index.search(np.array([query_embedding]), k=k)
    #     return [self.texts[i] for i in I[0]]

        # Ensure the query_embedding is a 2D array
        query_embedding = np.array([query_embedding], dtype=np.float32)  # Reshape to (1, dimension)

        # Perform the search
        D, I = self.index.search(query_embedding, k)  # Correctly call search

        # Return the relevant documents
        return [self.texts[i] for i in I[0]]
    def preprocess_question(self, question):
        """Preprocess the user question for clarity and specificity"""
        # Check if the question contains location-related keywords
        location_keywords = ['where', 'city', 'region', 'place', 'location']
        if not any(keyword in question.lower() for keyword in location_keywords):
            question += " Please specify the location or city you're asking about."
        return question

    def generate_prompt(self, query, relevant_docs):
        """Create prompt for the LLM"""
        context1 = "\n".join(relevant_docs)
        prompt = f"""Use the following pieces of context to create a structured travel itinerary.

        # INSTRUCTIONS
        1. If the user provides a specific destination, create a detailed itinerary for that location.
        2. If the user does not specify a destination, suggest one region in Saudi Arabia that is particularly great for the activity mentioned in the query. 
        In this case, do not provide a full itinerary but ask the user to choose a region based on your suggestions.
        3. Ensure that all activities are tied to a single destination and that they align with the user's interests as expressed in the query.
        4. If you cannot find enough information in the context to answer the question, respond with: "I don’t have enough information to answer this question."
            
        # IMPORTANT
        If user doesn't provide destination DO NOT GIVE ITENERARY, but suggest 1 region that is really great based on 
        f"Good regions in Saudi Arabia for {query} given {context1}"
        IN THIS CASE IGNORE INSTRUCTIONS BELOW and ask user to provide region based on your suggestions

        Please provide the itinerary in the following format:
        
        Itinerary:
        - Destination: [Destination Name]
          - Activities: [List of Activities]
          - Additional Notes: [Any other relevant information]
        
        Context:
        {context1}
        Question: {query}
        Answer:"""
        return prompt

    def format_response(self, response):
        """Format the response into a structured itinerary"""
        # Here you can implement any additional formatting logic if needed
        return response.strip()

    def query(self, question):
        """Full RAG pipeline"""
        # Preprocess the user question
        question = self.preprocess_question(question)
        # 1. Retrieve relevant documents
        relevant_docs = self.retrieve(question)
        # 2. Generate prompt with context
        prompt = self.generate_prompt(question, relevant_docs)
        # 3. Get answer from LLM
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates structured travel itineraries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens
        )
        # Format the response
        structured_itinerary = self.format_response(response.choices[0].message.content)
        return structured_itinerary