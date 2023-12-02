import streamlit as st


def main():
    st.set_page_config(page_title="Chat with PDFs!", page_icon=":books")

    st.header("Chat with PDFs! :books:")
    st.text_input("Ask a Question!:")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Put your files here and click on Process!")
        st.button("Process")


if __name__ == ' __main__':
    main()
