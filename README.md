# LangGraph Agentic AI - Stateful AI Agent Framework

A sophisticated multi-agent AI framework built with LangGraph that creates stateful, tool-integrated conversational agents with dynamic graph-based workflow orchestration, featuring both basic chatbot and advanced tool-enabled agent capabilities.

## 🎯 Project Overview

This project demonstrates advanced AI agent architecture using LangGraph for building stateful, multi-modal conversational systems. The framework supports multiple agent types with tool integration, state management, and dynamic workflow orchestration, showcasing modern approaches to autonomous AI agent development and deployment.

## 🚀 Key Features

### Agent Architecture
- **Stateful Agents**: Persistent conversation state management
- **Multi-Modal Support**: Basic chatbot and tool-integrated agents
- **Graph-Based Orchestration**: LangGraph workflow management
- **Dynamic Tool Binding**: Runtime tool integration and execution
- **Conditional Logic**: Smart agent routing and decision making

### LLM Integration
- **Groq LLM Support**: High-performance language model integration
- **Multiple Model Options**: Llama3-8B and Llama3-70B variants
- **API Key Management**: Secure credential handling
- **Model Configuration**: Dynamic model selection and tuning

### Tool Integration
- **Tavily Search**: Real-time web search capabilities
- **Tool Node Architecture**: Modular tool integration framework
- **Conditional Tool Execution**: Smart tool routing based on context
- **Tool Result Processing**: Structured tool output handling

### Web Interface
- **Streamlit Dashboard**: Interactive web-based interface
- **Real-time Chat**: Live conversation with AI agents
- **Configuration Panel**: Dynamic agent and model selection
- **Session Management**: Persistent conversation state

## 🏗️ Architecture Overview

### Agent Graph Structure
```
START → Chatbot Node → Tool Decision → Tool Node → Response
  ↓         ↓             ↓            ↓         ↓
State   Process      Conditional    Execute   Update
Mgmt    Messages     Routing        Tools     State
```

### Component Architecture
```
LangGraphAgenticAI/
├── src/langgraphagenticai/
│   ├── main.py                    # Application entry point
│   ├── LLMS/
│   │   └── groqllm.py            # Groq LLM integration
│   ├── graph/
│   │   └── graph_builder.py      # Graph construction logic
│   ├── nodes/
│   │   ├── basic_chatbot_node.py # Basic conversation node
│   │   └── chatbot_with_Tool_node.py # Tool-enabled node
│   ├── state/
│   │   └── state.py              # State management
│   ├── tools/
│   │   └── search_tool.py        # Tool integration
│   ├── ui/
│   │   └── streamlitui/          # Web interface
│   └── vectorstore/              # Vector database (future)
├── app.py                        # Streamlit application
└── requirements.txt              # Dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Streamlit
- LangChain & LangGraph
- Groq API Key
- Tavily API Key (for tool integration)

### 1. Clone Repository
```bash
git clone <repository-url>
cd langgraph-agentic-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
- **Groq API**: Get from [Groq Console](https://console.groq.com/keys)
- **Tavily API**: Get from [Tavily Dashboard](https://app.tavily.com/home)

### 5. Run Application
```bash
streamlit run app.py
```

### 6. Access Interface
- **URL**: http://localhost:8501
- **Interface**: Configure agent → Select tools → Start conversation

## 🤖 Agent Types & Capabilities

### 1. Basic Chatbot Agent

#### Node Implementation
```python
class BasicChatbotNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        return {"messages": self.llm.invoke(state['messages'])}
```

#### Graph Configuration
```python
def basic_chatbot_build_graph(self):
    self.basic_chatbot_node = BasicChatbotNode(self.llm)
    self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
    self.graph_builder.add_edge(START, "chatbot")
    self.graph_builder.add_edge("chatbot", END)
```

### 2. Tool-Integrated Agent

#### Advanced Node Implementation
```python
class ChatbotWithToolNode:
    def create_chatbot(self, tools):
        llm_with_tools = self.llm.bind_tools(tools)
        
        def chatbot_node(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        return chatbot_node
```

#### Graph with Conditional Logic
```python
def chatbot_with_tools_build_graph(self):
    tools = get_tools()
    tool_node = create_tool_node(tools)
    
    self.graph_builder.add_node("chatbot", chatbot_node)
    self.graph_builder.add_node("tools", tool_node)
    
    self.graph_builder.add_edge(START, "chatbot")
    self.graph_builder.add_conditional_edges("chatbot", tools_condition)
    self.graph_builder.add_edge("tools", "chatbot")
```

## 🔧 Technical Implementation

### State Management
```python
class State(TypedDict):
    """Represents the structure of the state used in the graph."""
    messages: Annotated[list, add_messages]
```

### LLM Configuration
```python
class GroqLLM:
    def get_llm_model(self):
        groq_api_key = self.user_controls_input['GROQ_API_KEY']
        selected_groq_model = self.user_controls_input['selected_groq_model']
        
        llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)
        return llm
```

### Tool Integration
```python
def get_tools():
    """Return the list of tools to be used in the chatbot"""
    tools = [TavilySearchResults(max_results=2)]
    return tools

def create_tool_node(tools):
    """Creates and returns a tool node for the graph"""
    return ToolNode(tools=tools)
```

## 🌐 Streamlit Interface

### Configuration Panel
- **LLM Selection**: Choose between available language models
- **Model Variants**: Select specific model sizes (8B/70B parameters)
- **API Management**: Secure key input with validation
- **Use Case Selection**: Choose agent type and capabilities

### Chat Interface
```python
class DisplayResultStreamlit:
    def display_result_on_ui(self):
        if usecase == "Basic Chatbot":
            for event in graph.stream({'messages': ("user", user_message)}):
                with st.chat_message("user"):
                    st.write(user_message)
                with st.chat_message("assistant"):
                    st.write(value["messages"].content)
```

### Real-time Updates
- **Message Streaming**: Live conversation display
- **Tool Execution Visualization**: Real-time tool call tracking
- **State Persistence**: Conversation history maintenance
- **Error Handling**: Comprehensive user feedback

## 📊 Graph Orchestration

### Conditional Routing
```python
# Tool decision logic
self.graph_builder.add_conditional_edges("chatbot", tools_condition)
```

### Message Flow
1. **User Input** → State Update
2. **Chatbot Processing** → LLM Inference
3. **Tool Decision** → Conditional Logic
4. **Tool Execution** → External API Calls
5. **Response Generation** → Final Output

### State Transitions
```
Initial State → Message Processing → Tool Evaluation → Response Generation → Updated State
```

## 🔍 Advanced Features

### Dynamic Tool Binding
```python
llm_with_tools = self.llm.bind_tools(tools)
```

### Message Type Handling
```python
for message in res['messages']:
    if type(message) == HumanMessage:
        # Handle user messages
    elif type(message) == ToolMessage:
        # Handle tool responses
    elif type(message) == AIMessage:
        # Handle AI responses
```

### Configuration Management
```python
class Config:
    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS").split(", ")
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Hugging Face Spaces
```yaml
title: LanggraphAgenticAI
emoji: 🐨
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration
- **AWS/GCP**: Container deployment
- **Azure**: App Service deployment

## 🔄 Extension Capabilities

### Additional LLM Support
```python
# OpenAI Integration
from langchain_openai import ChatOpenAI

class OpenAILLM:
    def get_llm_model(self):
        return ChatOpenAI(api_key=self.api_key, model=self.model)
```

### Custom Tool Development
```python
from langchain.tools import Tool

def create_custom_tool():
    return Tool(
        name="CustomTool",
        description="Custom tool description",
        func=custom_function
    )
```

### Vector Store Integration
```python
# Future enhancement placeholder
from langchain_community.vectorstores import FAISS

class VectorStoreNode:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def similarity_search(self, query):
        return self.vectorstore.similarity_search(query)
```

## 📈 Performance & Scalability

### Optimization Features
- **Async Processing**: Non-blocking tool execution
- **State Caching**: Efficient memory management
- **Model Switching**: Dynamic LLM selection
- **Tool Parallelization**: Concurrent tool execution

### Monitoring & Analytics
- **Conversation Tracking**: Session analytics
- **Tool Usage Metrics**: Performance monitoring
- **Error Logging**: Comprehensive debugging
- **Response Time Tracking**: Performance optimization

## 🧪 Testing & Quality Assurance

### Test Coverage
```python
# Example test structure
def test_basic_chatbot_node():
    node = BasicChatbotNode(mock_llm)
    result = node.process(test_state)
    assert "messages" in result

def test_tool_integration():
    tools = get_tools()
    assert len(tools) > 0
    assert isinstance(tools[0], TavilySearchResults)
```

### Integration Testing
- **Graph Execution**: End-to-end workflow testing
- **Tool Functionality**: External API integration tests
- **State Management**: Persistence and retrieval validation
- **UI Components**: Interface interaction testing

## 🔮 Future Enhancements

### Advanced Agent Types
- **Multi-Agent Systems**: Agent collaboration frameworks
- **Specialized Agents**: Domain-specific agent development
- **Agent Memory**: Long-term conversation memory
- **Agent Learning**: Adaptive behavior improvement

### Enterprise Features
- **Authentication**: User management and access control
- **Multi-tenancy**: Isolated agent environments
- **Analytics Dashboard**: Comprehensive usage analytics
- **API Gateway**: RESTful agent interaction endpoints

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Implement agent improvements
4. Add comprehensive tests
5. Update documentation
6. Commit changes (`git commit -m 'Add enhancement'`)
7. Push to branch (`git push origin feature/enhancement`)
8. Create Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 👤 Author

**Ayush Poddar**
- Email: ayushpoddar351@gmail.com
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- **LangChain Team**: Foundational framework and tools
- **LangGraph Community**: Graph-based agent orchestration
- **Groq**: High-performance language model inference
- **Tavily**: Real-time search integration
- **Streamlit**: Interactive web application framework
- **AI Research Community**: Agent architecture innovations

## 📚 Key Learning Outcomes

This project demonstrates:
- **Advanced AI Agent Architecture**: Multi-agent system design and implementation
- **Graph-Based Orchestration**: Complex workflow management with LangGraph
- **Tool Integration**: External API integration and tool management
- **State Management**: Persistent conversation and context handling
- **Web Application Development**: Interactive AI interface creation
- **Modular Design**: Scalable, maintainable agent framework
- **Real-time Processing**: Live agent interaction and response handling

---

*This project showcases cutting-edge AI agent development using LangGraph for stateful, tool-integrated conversational systems, demonstrating modern approaches to autonomous agent architecture and deployment.*
