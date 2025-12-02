import streamlit as st
from qdrant_client import QdrantClient
from openai import OpenAI

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Yoga Institute Assistant", page_icon="🧘", layout="centered")

# ---------------------------
# Initialize Clients
# ---------------------------
@st.cache_resource
def init_clients():
    # Get API keys from Streamlit secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    qdrant_url = st.secrets.get("QDRANT_URL")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    
    if not openai_api_key or not qdrant_url or not qdrant_api_key:
        st.error("⚠️ API keys not configured. Please set up secrets in Streamlit Cloud or .streamlit/secrets.toml")
        st.stop()
    
    client = OpenAI(api_key=openai_api_key)
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )
    return client, qdrant

client, qdrant = init_clients()

# ---------------------------
# RAG Functions
# ---------------------------
def retrieve_chunks(query: str, top_k=4):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = resp.data[0].embedding
    
    results = qdrant.query_points(
        collection_name="Institutes",
        query=query_vector,
        limit=top_k
    )
    return results.points

def build_context(results):
    context = ""
    for r in results:
        if "content" in r.payload:
            context += r.payload["content"] + "\n"
    return context

def is_greeting_or_general(query):
    """Check if the query is a greeting or general conversation"""
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                 "greetings", "howdy", "what's up", "whats up", "sup"]
    query_lower = query.lower().strip()
    
    # Check for exact greetings or very short queries
    if query_lower in greetings or len(query.split()) <= 2 and any(g in query_lower for g in greetings):
        return True
    return False

def is_asking_for_institutes_list(query):
    """Check if user is asking for list of institutes"""
    keywords = ["list", "all institutes", "certified", "verified", "which institutes", 
                "show institutes", "available institutes", "what institutes"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

def get_all_institutes():
    """Retrieve all unique institutes from the database"""
    # Query with a generic embedding to get diverse results
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="yoga institute information"
    )
    query_vector = resp.data[0].embedding
    
    results = qdrant.query_points(
        collection_name="Institutes",
        query=query_vector,
        limit=50  # Get more results to find all institutes
    )
    
    institutes = set()
    for r in results.points:
        if "institute_name" in r.payload:
            institutes.add(r.payload["institute_name"])
        elif "content" in r.payload:
            # Try to extract institute name from content
            content = r.payload["content"]
            if "Athayog" in content:
                institutes.add("Athayog Living")
    
    return sorted(list(institutes))

def ask_rag(query, chat_history):
    usage_info = None
    
    # Handle greetings (only if it's the first message or no context)
    if is_greeting_or_general(query) and len(chat_history) == 0:
        return """Welcome to the Yoga AI Assistant! 🧘

I'm here to help you find information about yoga institutes, their classes, subscriptions, schedules, and more. 

You can ask me about:
- Specific yoga institutes and their offerings
- Class schedules and timings
- Subscription plans and pricing
- Location details
- Special programs and workshops

Feel free to ask me anything about our certified yoga institutes, or simply ask "What institutes are available?" to see the full list.

How may I assist you today?""", None
    
    # Handle request for institutes list
    if is_asking_for_institutes_list(query):
        institutes = get_all_institutes()
        if institutes:
            institutes_list = "\n".join([f"- {inst}" for inst in institutes])
            return f"""Here are the certified and verified yoga institutes in our database:

{institutes_list}

These institutes are verified and offer professional yoga instruction. You can ask me specific questions about any of these institutes, such as their class schedules, subscription plans, locations, or special programs.

Which institute would you like to know more about?""", None
        else:
            return "I'm currently updating our institute database. Please try again in a moment, or ask me about a specific institute you're interested in.", None
    
    # Handle specific queries with RAG
    results = retrieve_chunks(query)
    context = build_context(results)
    
    # Check if we have relevant context
    if not context.strip():
        return """I don't have specific information about that in my database. 

I can help you with information about our certified yoga institutes, including:
- Class schedules and timings
- Subscription plans and pricing
- Location details
- Special programs

You can ask "What institutes are available?" to see all certified institutes, or ask me about a specific institute you're interested in.""", None
    
    # Build system message with context
    system_message = f"""You are a professional Yoga AI Assistant for certified yoga institutes. Your role is to provide accurate, helpful, and professional information.

IMPORTANT INSTRUCTIONS:
1. Answer questions ONLY about the specific institute mentioned in the user's query
2. Use ONLY the information provided in the context below
3. Be professional, clear, and concise
4. If the context doesn't contain information about the specific institute asked, politely say so
5. Format pricing and schedules clearly
6. Always maintain a helpful and welcoming tone
7. Remember the conversation history and provide contextual responses

Context from database:
{context}"""
    
    # Build messages array with conversation history
    messages = [{"role": "system", "content": system_message}]
    
    # Add chat history (limit to last 10 messages to avoid token limits)
    for msg in chat_history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Extract usage information
    usage = response.usage
    usage_info = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }
    
    return response.choices[0].message.content, usage_info

def calculate_cost(usage_info):
    """Calculate cost based on GPT-4o-mini pricing"""
    if not usage_info:
        return None
    
    # GPT-4o-mini pricing (as of Dec 2024)
    # Input: $0.150 per 1M tokens
    # Output: $0.600 per 1M tokens
    input_cost = (usage_info["prompt_tokens"] / 1_000_000) * 0.150
    output_cost = (usage_info["completion_tokens"] / 1_000_000) * 0.600
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧘 Yoga Institute Assistant")
st.markdown("Ask me anything about yoga classes, subscriptions, and schedules!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# Sidebar for usage statistics and controls
with st.sidebar:
    st.header("📊 Usage Statistics")
    st.metric("Total Tokens Used", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.6f}")
    st.caption("GPT-4o-mini pricing: $0.150/1M input tokens, $0.600/1M output tokens")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset Stats"):
            st.session_state.total_cost = 0.0
            st.session_state.total_tokens = 0
            st.rerun()
    
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show usage info if available
        if message["role"] == "assistant" and "usage" in message and message["usage"]:
            with st.expander("📈 Token Usage"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Input", message["usage"]["prompt_tokens"])
                col2.metric("Output", message["usage"]["completion_tokens"])
                col3.metric("Total", message["usage"]["total_tokens"])
                if "cost" in message:
                    st.caption(f"Cost: ${message['cost']['total_cost']:.6f}")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get assistant response with full chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass chat history (excluding the current message we just added)
            response, usage_info = ask_rag(prompt, st.session_state.messages[:-1])
            st.markdown(response)
            
            # Display and store usage info
            if usage_info:
                cost_info = calculate_cost(usage_info)
                
                # Update totals
                st.session_state.total_tokens += usage_info["total_tokens"]
                st.session_state.total_cost += cost_info["total_cost"]
                
                with st.expander("📈 Token Usage"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input", usage_info["prompt_tokens"])
                    col2.metric("Output", usage_info["completion_tokens"])
                    col3.metric("Total", usage_info["total_tokens"])
                    st.caption(f"Cost: ${cost_info['total_cost']:.6f}")
                
                # Add to chat history with usage info
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "usage": usage_info,
                    "cost": cost_info
                })
            else:
                # Add to chat history without usage info
                st.session_state.messages.append({"role": "assistant", "content": response})
