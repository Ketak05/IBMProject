import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import uuid
import traceback
import random

# Set up the Streamlit page
st.set_page_config(
    page_title="AI Story Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Remove horizontal spacing and center content */
    .stApp {
        max-width: 100%;
        padding: 0 2rem;
        margin: 0 auto;
        font-size: 20px; /* Increased global font size */
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #6f42c1, #20c997);
        padding: 40px 30px;
        border-radius: 15px;
        margin-bottom: 35px;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .header-container h1 {
        font-size: 3rem !important;
        margin-bottom: 10px;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .header-container p {
        font-size: 1.5rem !important;
        opacity: 0.9;
    }

    /* Story container improvements */
    .story-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 35px;
        border: 1px solid #dee2e6;
        margin-top: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .story-text {
        font-family: 'Georgia', serif;
        font-size: 22px; /* Larger font */
        line-height: 1.8;
        color: #212529;
        white-space: pre-wrap;
    }

    /* Character and theme badges */
    .character-badge {
        background-color: #e2d4f8;
        color: #6f42c1;
        border-radius: 20px;
        padding: 8px 15px;
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        font-size: 1rem;
        font-weight: 500;
    }

    .theme-badge {
        background-color: #d1f7eb;
        color: #20c997;
        border-radius: 20px;
        padding: 8px 15px;
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .mood-badge {
        background-color: #ffe8cc;
        color: #fd7e14;
        border-radius: 20px;
        padding: 8px 15px;
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        font-size: 1rem;
        font-weight: 500;
    }

    /* Label and input improvements */
    label {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    .stTextInput input, .stTextArea textarea {
        font-size: 1.1rem !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        font-size: 1.2rem !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        gap: 25px !important;
    }
    
    .stRadio label {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* Section headers */
    h3 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
        color: #343a40;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-width: 5px !important;
    }
    
    /* Animation for generated stories */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .story-animation {
        animation: fadeIn 1s ease-out;
    }
    
    /* Story title styling */
    .story-title {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #343a40;
        margin-bottom: 15px !important;
        font-family: 'Georgia', serif;
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background-color: #20c997 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 10px !important;
    }
    
    /* Divider styling */
    hr {
        margin: 30px 0 !important;
        border-color: #dee2e6 !important;
    }
    
    /* Caption styling */
    .stCaption {
        font-size: 1rem !important;
        color: #6c757d !important;
    }
    
    /* Custom styles for mood selection */
    .mood-selector {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    
    .mood-option {
        padding: 10px 20px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .mood-option:hover {
        background-color: #e9ecef;
    }
    
    .mood-option.selected {
        background-color: #ffe8cc;
        border-color: #fd7e14;
        color: #fd7e14;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Title and description in a custom header
st.markdown("""
<div class="header-container">
    <h1>✨ AI Story Generator ✨</h1>
    <p>Create unique, imaginative stories with advanced language models</p>
</div>
""", unsafe_allow_html=True)

# Function to initialize HuggingFace LLM
def initialize_huggingface_llm(api_key, model_name, temperature=0.7, max_length=None):
    """Initialize the HuggingFace LLM"""
    # Set max_length based on story length if not specified
    if max_length is None:
        max_length = 2000  # Default
    
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        max_new_tokens=max_length,
        provider=selected_provider
    )
    return llm

# Function to create story prompt
def setup_story_prompt(characters, theme, plot, length, mood, genre=None, setting=None):
    """Create a prompt for story generation based on user parameters"""
    template = """
    You are a creative story writer. Write an engaging and imaginative story based on 
    the characters, theme, plot outline, and requested length provided by the user.
    
    Guidelines:
    - Create vivid, memorable characters with distinct personalities
    - Maintain consistent tone and pacing appropriate to the theme
    - Include descriptive language for settings and characters
    - Use natural dialogue that reveals character and advances the plot
    - Structure the story with a clear beginning, middle, and end
    - Match the requested length (short ~500 words, medium ~1500 words, long ~3000 words)
    - Maintain the requested mood throughout the story
    
    Please write a {length} story with the following elements:
    
    Characters: {characters}
    Theme: {theme}
    Plot: {plot}
    Mood: {mood}
    {genre_text}
    {setting_text}
    
    Make it engaging, creative, and coherent. Ensure the story has a proper ending and is complete.
    Return only the story without introductions or explanations.
    """
    
    # Add optional parameters if provided
    genre_text = f"Genre: {genre}" if genre else ""
    setting_text = f"Setting: {setting}" if setting else ""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["characters", "theme", "plot", "length", "mood", "genre_text", "setting_text"]
    )
    
    return prompt

# Function to generate story
def generate_story(api_key, characters, theme, plot, length, mood, temperature, model_name, genre=None, setting=None):
    """Generate a story using the HuggingFace LLM"""
    # Define length tokens
    length_tokens = {
        "short": 1000,   # ~500 words
        "medium": 2500,  # ~1250 words
        "long": 4500     # ~2250 words
    }
    
    max_tokens = length_tokens.get(length, 2500)
    
    # Initialize the LLM
    llm = initialize_huggingface_llm(
        api_key,
        model_name,
        temperature,
        max_tokens
    )
    
    # Create and format the prompt
    prompt = setup_story_prompt(characters, theme, plot, length, mood, genre, setting)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the story
    result = llm_chain.invoke({
        "characters": characters,
        "theme": theme,
        "plot": plot,
        "length": length,
        "mood": mood,
        "genre_text": f"Genre: {genre}" if genre else "",
        "setting_text": f"Setting: {setting}" if setting else ""
    })
    
    # Extract the story text
    if isinstance(result, dict) and "text" in result:
        story_text = result["text"]
    else:
        story_text = str(result)
    
    return story_text

# Function to save story
def save_story(story_text, title="My Story"):
    """Save story to a file and provide download link"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(story_text)
        temp_path = temp_file.name
    
    # Read the file for download
    with open(temp_path, 'rb') as f:
        st.download_button(
            label="⬇️ Download Story",
            data=f,
            file_name=f"{title.replace(' ', '_')}.txt",
            mime="text/plain"
        )

# Generate a random title suggestion
def generate_title_suggestion(theme=None, characters=None):
    prefix_options = ["The", "A", "Chronicles of", "Tales of", "Journey to", "Secrets of", "Beyond the", "Whispers of"]
    middle_options = ["Adventure", "Mystery", "Quest", "Legacy", "Secret", "Dream", "Shadow", "Light", "Echo"]
    suffix_options = ["Revealed", "Untold", "Awakened", "Discovered", "Beyond", "Within", "Forgotten"]
    
    if theme:
        # Try to incorporate the theme
        words = theme.split()
        if words:
            middle_options.append(words[0].title())
    
    if characters:
        # Try to incorporate a character
        char_words = characters.split(',')[0].strip() if ',' in characters else characters.strip()
        if char_words:
            if ' ' in char_words:
                char_words = char_words.split(' ')[0]
            middle_options.append(char_words.title())
    
    title_pattern = random.choice([
        f"{random.choice(prefix_options)} {random.choice(middle_options)}",
        f"{random.choice(prefix_options)} {random.choice(middle_options)} {random.choice(suffix_options)}",
        f"{random.choice(middle_options)} of {random.choice(suffix_options)}"
    ])
    
    return title_pattern

# Session state initialization
if 'generated_story' not in st.session_state:
    st.session_state['generated_story'] = None
if 'story_metadata' not in st.session_state:
    st.session_state['story_metadata'] = None
if 'selected_mood' not in st.session_state:
    st.session_state['selected_mood'] = "adventurous"
if 'title_suggestion' not in st.session_state:
    st.session_state['title_suggestion'] = "The Adventure Begins"

# Sidebar for API configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API key input
    api_key = st.text_input("HuggingFace API Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    st.markdown("---")
    st.markdown("## 🤖 Model Settings")
    
    # Model selection
    model_options = {
        "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "UIGEN-T2-7B": "Tesslate/UIGEN-T2-7B",
        "QwQ-32B": "Qwen/QwQ-32B",
        "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "Nous-Hermes-2-Mixtral-8x7B-DPO": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "Falcon-40B-Instruct": "tiiuae/falcon-40b-instruct",
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    }
    selected_model = st.selectbox("Select Model", list(model_options.keys()))
    model_name = model_options[selected_model]
    
    # Generation parameters
    st.markdown("### 🎯 Generation Parameters")
    temperature = st.slider("Temperature", 
                          min_value=0.1, 
                          max_value=1.0, 
                          value=0.75, 
                          step=0.05,
                          help="Higher values make the output more creative and random, lower values make it more deterministic")
    
    # Provider selection
    provider_options = [
        "auto",
        "hf-inference",     # HF Inference API
        "hyperbolic",
        "sambanova",
        "novita",
        "together",
        "cohere",
        "replicate",
        "fal",
        "fireworks",
        "nebius",
        "cerebras"
    ]
    selected_provider = st.selectbox("Select Provider", provider_options, index=0)
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Higher temperature values create more creative but potentially less coherent stories.")

# Story parameters section
st.header("📝 Story Parameters")

# Create columns for more compact layout
col1, col2 = st.columns(2)

with col1:
    characters = st.text_area("👥 Characters (comma separated)", 
                            placeholder="e.g., a curious scientist, a wise alien, a talking computer",
                            help="List the main characters of your story")

with col2:
    theme = st.text_input("🎭 Theme", 
                        placeholder="e.g., discovery, friendship, betrayal",
                        help="The central theme or mood of your story")

# Add genre and setting options
col3, col4 = st.columns(2)

with col3:
    genre = st.selectbox("📚 Genre", 
                       ["", "Fantasy", "Science Fiction", "Mystery", "Romance", "Horror", 
                        "Adventure", "Historical Fiction", "Thriller", "Comedy", "Drama", "Fairy Tale"],
                       index=0,
                       help="The literary genre of your story")

with col4:
    setting = st.text_input("🌍 Setting",
                          placeholder="e.g., medieval kingdom, space station, underwater city",
                          help="Where and when your story takes place")

plot = st.text_area("📋 Plot Outline", 
                  placeholder="e.g., A scientist discovers an alien artifact that allows communication with a distant civilization",
                  help="Brief description of the story's main events")

# Mood selector with visual buttons
st.markdown("### 😊 Story Mood")
moods = ["adventurous", "mysterious", "romantic", "humorous", "melancholic", "suspenseful", "whimsical", "inspirational"]

# Create a 4x2 grid for mood options
mood_cols = st.columns(4)
for i, mood in enumerate(moods):
    col_idx = i % 4
    with mood_cols[col_idx]:
        if st.button(
            mood.title(), 
            key=f"mood_{mood}",
            help=f"Set the story mood to {mood}",
            use_container_width=True,
            type="secondary" if st.session_state['selected_mood'] != mood else "primary"
        ):
            st.session_state['selected_mood'] = mood
            
st.markdown(f"Selected mood: <span class='mood-badge'>{st.session_state['selected_mood'].title()}</span>", unsafe_allow_html=True)

# Length selection with radio buttons
st.markdown("### 📏 Story Length")
length = st.radio("Select Length", 
                ["short", "medium", "long"],
                index=1,
                horizontal=True,
                help="Short ~500 words, Medium ~1250 words, Long ~2250 words")

# Generate button
st.markdown("### 🚀 Generate Your Story")
generate_button = st.button("✨ Create My Story ✨", type="primary", use_container_width=True)

# Display previous story if available
if st.session_state['generated_story'] and not generate_button:
    st.markdown("<div class='story-animation'>", unsafe_allow_html=True)
    
    # Display metadata
    st.markdown("## 📖 Your Generated Story")
    
    meta = st.session_state['story_metadata']
    
    # Display title input
    story_title = st.text_input("Story Title", value=st.session_state.get('title_suggestion', "My Story"))
    
    st.markdown(f"<h3 class='story-title'>{story_title}</h3>", unsafe_allow_html=True)
    
    # Create tabs for Story and Metadata
    story_tab, meta_tab = st.tabs(["📖 Story", "ℹ️ Story Details"])
    
    with story_tab:
        # Display story in a styled container
        st.markdown("<div class='story-container'><div class='story-text'>" + 
                    st.session_state['generated_story'].replace('\n', '<br>') + 
                    "</div></div>", unsafe_allow_html=True)
        
        # Save story option
        save_story(st.session_state['generated_story'], story_title)
    
    with meta_tab:
        # Display metadata in a more organized way
        meta_col1, meta_col2 = st.columns(2)
        
        with meta_col1:
            st.markdown("#### Theme & Mood")
            st.markdown(f"**Theme:** <span class='theme-badge'>{meta['theme']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Mood:** <span class='mood-badge'>{meta['mood'].title()}</span>", unsafe_allow_html=True)
            if 'genre' in meta and meta['genre']:
                st.markdown(f"**Genre:** {meta['genre']}")
            st.markdown(f"**Length:** {meta['length'].title()} (~{2500 if meta['length'] == 'medium' else 1000 if meta['length'] == 'short' else 4500} tokens)")
        
        with meta_col2:
            st.markdown("#### Characters")
            chars_html = ""
            for char in meta['characters'].split(','):
                if char.strip():
                    chars_html += f"<span class='character-badge'>{char.strip()}</span> "
            st.markdown(f"{chars_html}", unsafe_allow_html=True)
            
            if 'setting' in meta and meta['setting']:
                st.markdown("#### Setting")
                st.markdown(f"{meta['setting']}")
            
        st.markdown("#### Plot Outline")
        st.markdown(f"{meta['plot']}")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add option to regenerate with same parameters
    if st.button("🔄 Regenerate with Same Parameters", use_container_width=True):
        with st.spinner(f"Regenerating your {meta['length']} story..."):
            try:
                # Generate new story with same parameters
                story = generate_story(
                    api_key,
                    meta['characters'],
                    meta['theme'],
                    meta['plot'],
                    meta['length'],
                    meta['mood'],
                    temperature,
                    model_name,
                    meta.get('genre', ''),
                    meta.get('setting', '')
                )
                
                # Store in session state
                st.session_state['generated_story'] = story
                
                # Generate a new title suggestion
                st.session_state['title_suggestion'] = generate_title_suggestion(meta['theme'], meta['characters'])
                
                # Force a rerun to display the story
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")
                st.info("Try adjusting your parameters or checking your API key.")
                st.expander("See error details").code(traceback.format_exc(), language='python')

# Generate story when button is clicked
if generate_button:
    # Validate inputs
    if not characters:
        characters = "a brave knight and a wise dragon"
    if not theme:
        theme = "adventure"
    if not plot:
        plot = "A journey to find a magical artifact"
    
    # Get current mood
    selected_mood = st.session_state['selected_mood']
    
    if not api_key:
        st.error("⚠️ Please enter your HuggingFace API token in the sidebar.")
    else:
        with st.spinner(f"✨ Crafting your {length} story with a {selected_mood} mood..."):
            try:
                # Generate story
                story = generate_story(
                    api_key,
                    characters,
                    theme,
                    plot,
                    length,
                    selected_mood,
                    temperature,
                    model_name,
                    genre,
                    setting
                )
                
                # Generate a title suggestion
                title_suggestion = generate_title_suggestion(theme, characters)
                st.session_state['title_suggestion'] = title_suggestion
                
                # Store in session state
                st.session_state['generated_story'] = story
                st.session_state['story_metadata'] = {
                    'characters': characters,
                    'theme': theme,
                    'plot': plot,
                    'length': length,
                    'mood': selected_mood,
                    'genre': genre,
                    'setting': setting
                }
                
                # Force a rerun to display the story
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")
                st.info("Try adjusting your parameters or checking your API key.")
                st.expander("See error details").code(traceback.format_exc(), language='python')

# Add a section for story prompts and inspiration
st.markdown("---")
with st.expander("🎯 Story Prompts & Inspiration"):
    st.markdown("""
    ### Need inspiration? Try these story prompts:
    
    1. **Mystery in the Abandoned Library**: Characters discover ancient books with strange properties in a forgotten library.
    
    2. **The Last of Their Kind**: Two beings from dying species must work together to survive in a changing world.
    
    3. **Time Loop Adventure**: A character keeps reliving the same day, but each time something crucial changes.
    
    4. **The Friendship Algorithm**: An AI and human form an unexpected friendship that challenges both their worldviews.
    
    5. **A Garden of Memories**: A magical garden where plants grow from people's memories faces a mysterious blight.
    
    6. **The Impossible Journey**: Travelers must cross a landscape that defies the laws of physics to reach their destination.
    
    7. **Between Two Worlds**: A character discovers they can travel between parallel universes but faces consequences for altering either.
    
    8. **The Object's History**: A seemingly ordinary object passes through many hands, changing each person's life in unexpected ways.
    """)

# Information section
with st.expander("ℹ️ About the Story Generator"):
    st.markdown("""
    ### How It Works
    
    This app uses HuggingFace language models to generate creative stories based on your input parameters:
    
    - **Characters**: The main figures in your story
    - **Theme**: The central feeling or concept of your story
    - **Genre**: The literary category of your story
    - **Setting**: Where and when your story takes place
    - **Plot**: A brief outline of what happens
    - **Mood**: The emotional tone of your story
    - **Length**: How long the story should be
    
    ### Length Options
    
    - **Short**: Approximately 500 words
    - **Medium**: Approximately 1250 words
    - **Long**: Approximately 2250 words
    
    ### Tips for Better Stories
    
    1. **Be specific** about your characters - give them traits, motivations, or backgrounds
    2. **Choose interesting themes** that allow for conflict or growth
    3. **Provide a clear plot outline** with a beginning, middle, and end
    4. **Select a mood** that complements your theme and plot
    5. **Adjust the temperature** to control creativity (higher = more creative but potentially less coherent)
    6. **Mix different genres** for unique story combinations
    """)

with st.expander("🤖 About the Models"):
    st.markdown("""
    ### HuggingFace Endpoint Models
    These models run on HuggingFace's servers via API calls. You need an API token to use them.
    
    - **Mistral-7B-Instruct-v0.3**: Instruction-tuned version of Mistral-7B, excellent for creative tasks.
    - **Mixtral-8x7B-Instruct-v0.1**: A mixture of experts model with strong creative writing capabilities.
    - **Llama-3.1-8B-Instruct**: Meta's Llama 3.1 model with 8B parameters, good balance of performance and speed.
    - **Falcon-40B-Instruct**: Large language model from TII with 40B parameters, great for detailed stories.
    - **Qwen2.5-7B-Instruct**: Alibaba's instruction-tuned model with strong narrative abilities.
    - **Nous-Hermes-2-Mixtral-8x7B-DPO**: Fine-tuned model with excellent creative writing capabilities.
    
    ### Generation Parameters
    - **Temperature**: Controls randomness and creativity. Higher values (e.g., 0.8) make output more creative and diverse, lower values (e.g., 0.2) make it more deterministic and focused.
    - **Provider**: The inference provider to use. "hf-inference" is recommended for most users.
    """)

with st.expander("⚠️ Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    
    1. **Provider errors**: If you get a provider error, try selecting "auto" or "hf-inference" from the provider dropdown.
    
    2. **API key**: Make sure your HuggingFace API token has the necessary permissions and is entered correctly.
    
    3. **Model access**: Some models require explicit approval on HuggingFace before you can use them via API.
    
    4. **Rate limits**: Free API tokens have usage limitations. Consider upgrading if you hit rate limits.
    
    5. **Incomplete stories**: If stories are being cut off, try using a shorter length setting or a model with higher token limits.
    
    6. **Slow generation**: Larger models (like Falcon-40B) may take longer to generate stories. Be patient or try a smaller model.
    
    7. **Content moderation**: Some models have content filters that may affect generation if your prompt contains sensitive topics.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
        <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 5px;">✨ AI Story Generator</p>
        <p style="color: #6c757d; font-size: 1rem;">Built with Streamlit, LangChain, and HuggingFace 🤗</p>
    </div>
    <div>
        <p style="color: #6c757d; font-size: 1rem;">v2.0.0 | Updated May 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)
