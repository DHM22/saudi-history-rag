import streamlit as st

# Page title
st.title("🎈 Simple Streamlit App")

# Text content
st.write("Welcome to this simple Streamlit example!")

# Header
st.header("Interactive Elements")

# Text input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! 👋")

# Slider
age = st.slider("Select your age:", 0, 100, 25)
st.write(f"You are {age} years old")

# Selectbox
option = st.selectbox(
    "Choose your favorite color:",
    ["Blue", "Red", "Green", "Yellow"]
)
st.write(f"You selected: {option}")

# Button
if st.button("Click me!"):
    st.success("Button clicked! 🎉")

# Display some data
st.header("Sample Data")
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 28],
    "City": ["NYC", "LA", "Chicago"]
}
st.dataframe(data)

