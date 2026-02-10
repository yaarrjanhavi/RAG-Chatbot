# RAG Chatbot - Healthcare Review Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Google's Gemini AI to answer questions about healthcare facility reviews. The system uses vector embeddings to retrieve relevant reviews and generates contextual responses using a large language model.

## 🌟 Features

- **Intelligent Review Analysis**: Leverages RAG architecture to provide accurate, context-aware answers from healthcare facility reviews
- **Vector Database**: Uses Chroma DB for efficient semantic search and document retrieval
- **Google Gemini Integration**: Powered by Gemini 2.5 Flash for fast and accurate responses
- **Interactive UI**: Built with Gradio for an easy-to-use chat interface
- **Batch Processing**: Handles large datasets with rate-limited batch processing
- **Customizable Retrieval**: Configurable top-k retrieval for optimal context

## 🏗️ Architecture

The system follows a standard RAG pipeline:

1. **Document Loading**: Reviews are loaded from CSV files
2. **Text Splitting**: Documents are chunked into manageable pieces using RecursiveCharacterTextSplitter
3. **Embedding**: Text chunks are converted to vector embeddings using Google Generative AI embeddings
4. **Vector Storage**: Embeddings are stored in Chroma DB for efficient retrieval
5. **Retrieval**: Relevant reviews are fetched based on user queries
6. **Generation**: Google Gemini generates responses using retrieved context

## 📋 Prerequisites

- Python 3.8+
- Google Colab (recommended) or local Python environment
- Google API Key for Gemini AI

## 🚀 Installation

1. Install required packages:

```bash
pip install --upgrade langchain
pip install --upgrade langchain-core
pip install --upgrade langchain-community
pip install --upgrade langchain-google-genai
pip install gradio
```

2. Set up your Google API key in Google Colab:
   - Store your API key in Colab's secrets as `google_api_key`
   - Or set it as an environment variable

## 💾 Data Setup

The project expects review data in CSV format. Update the following paths in the notebook:

```python
# Path to your review CSV files
data_dir = "/content/drive/MyDrive/Healthcare_Reviews/"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Vector database storage path
vector_db_path = "/content/drive/MyDrive/Healthcare_Reviews/chroma_db"
```

### Data Format

Your CSV should contain review data with columns that can be loaded using LangChain's CSVLoader.

## 🔧 Configuration

### Model Parameters

```python
# Initialize the Chat Model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  # Adjust for more/less creative responses
    google_api_key=userdata.get('google_api_key')
)
```

### Text Splitting

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size as needed
    chunk_overlap=200,    # Overlap between chunks
    length_function=len,
    is_separator_regex=False
)
```

### Retrieval Settings

```python
reviews_retriever = reviews_vector_db.as_retriever(k=10)  # Number of reviews to retrieve
```

## 📖 Usage

### Running the Chatbot

1. **Load and Process Data**: Run the data loading and vector database creation cells
2. **Launch the Interface**: Execute the Gradio interface cell
3. **Ask Questions**: Interact with the chatbot through the web interface

### Example Queries

- "Has anyone complained about communication with the hospital staff?"
- "What do patients say about the cleanliness of the facility?"
- "Are there any reviews mentioning wait times?"
- "What are the most common positive feedback themes?"

### Programmatic Usage

```python
# Direct query without UI
question = "Has anyone complained about communication with the hospital staff?"
response = review_chain.invoke(question)
print(response)
```

## 🎯 How It Works

### 1. System Prompt Template

The chatbot uses a carefully crafted prompt template:

```python
review_prompt_template = ChatPromptTemplate.from_template(
    """You're an assistant that helps users understand patient reviews...
    
    Reviews:
    {context}
    
    Question: {question}
    """
)
```

### 2. RAG Chain

The chain processes queries through multiple steps:

```python
review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
```

### 3. Response Generation

- Retrieves top 10 relevant reviews
- Formats them with the user question
- Generates contextual response using Gemini
- Parses and returns clean output

## 🔒 Rate Limiting

The vector database creation includes rate limiting to respect API quotas:

```python
time.sleep(30)  # 30-second pause between batches
```

Adjust this value based on your API tier and requirements.

## 🎨 User Interface

The Gradio interface provides:
- Clean chat-style interaction
- Message history
- Real-time responses
- Easy deployment (local or Colab)

```python
interface = gr.ChatInterface(
    fn=respond_to_user_question,
    title="Review Helper Bot"
)
interface.launch(debug=True)
```

## 📊 Performance Considerations

- **Chunk Size**: Smaller chunks = more precise retrieval, larger chunks = more context
- **Top-k Retrieval**: More documents = better context but slower responses
- **Temperature**: Lower values = more focused responses, higher values = more creative
- **Batch Size**: Adjust based on memory constraints and API rate limits

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Google API key is correctly set in Colab secrets
2. **Memory Issues**: Reduce batch size or chunk size for large datasets
3. **Rate Limiting**: Increase sleep time between batches if hitting API limits
4. **Vector DB Loading**: Ensure the correct path and that the database was created successfully

## 🔄 Future Enhancements

- [ ] Add conversation memory for multi-turn interactions
- [ ] Implement source attribution in responses
- [ ] Add support for multiple data sources
- [ ] Create a deployment-ready API endpoint
- [ ] Add evaluation metrics for response quality
- [ ] Implement caching for frequently asked questions

## 📚 Dependencies

- **LangChain**: Framework for LLM application development
- **LangChain Google GenAI**: Google Gemini integration
- **Chroma DB**: Vector database for embeddings
- **Gradio**: UI framework for ML applications
- **Google Generative AI**: Embedding and chat models

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project was developed in Google Colab and is optimized for that environment. Some modifications may be needed for local deployment.
