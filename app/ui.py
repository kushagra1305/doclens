"""
app/ui.py - Step 9 of DocLens: Streamlit front end.

Run (from the doclens/ folder, with the API already running):
    streamlit run app/ui.py
"""
import requests
import streamlit as st

st.set_page_config(page_title="DocLens", page_icon="📄", layout="centered")

MODELS = {
    "Accurate (qwen2.5:7b, slower)": "qwen2.5:7b",
    "Fast (llama3.2:3b, less reliable)": "llama3.2:3b",
}
EXAMPLES = [
    "For how many years can a business loss be carried forward?",
    "What does section 248 say about powers to requisition?",
    "Who is eligible to use ITR-1?",
    "What did the CBDT say about TDS certificate deadlines?",
    "What is the capital of France?",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "question" not in st.session_state:
    st.session_state.question = ""

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API address", "http://localhost:8000")
    model_label = st.radio("Model", list(MODELS))
    st.caption("The accurate model reads long passages and negations better, "
               "but takes about a minute per answer on a laptop.")
    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

st.title("📄 DocLens")
st.write("Ask a question about the Income-tax Act 2025, ITR-1 instructions and CBDT circulars. "
         "Every answer shows the file and page it came from, or says it doesn't know.")

question = st.text_input("Your question", key="question")
ask = st.button("Ask", type="primary")


def show_result(res: dict) -> None:
    if res["refused_by"]:
        st.info(res["answer"])
        st.caption(f"Refused by: {res['refused_by']} | best reranker score: {res['top_score']:.2f}")
        return
    st.success(res["answer"])
    if not res["cited_by_model"]:
        st.warning("The model did not cite its sources. These are the closest passages.")
    st.subheader("Sources")
    for s in res["sources"]:
        nums = ", ".join(f"[{n}]" for n in s["passages"])
        st.markdown(f"- {nums} **{s['source']}**, page **{s['page']}**")
    with st.expander("Show the source passages"):
        for p in res["passages"]:
            st.markdown(f"**[{p['n']}] {p['source']}, page {p['page']}**")
            st.text(p["text"])
    st.caption(f"Model: {res['model']} | best reranker score: {res['top_score']:.2f}")


if ask and question.strip():
    try:
        with st.spinner("Searching the documents and writing an answer..."):
            r = requests.post(
                f"{api_url}/ask",
                json={"question": question, "model": MODELS[model_label]},
                timeout=900,
            )
            r.raise_for_status()
        result = r.json()
        st.session_state.history.insert(0, result)
        show_result(result)
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Start it first with: uvicorn app.api:app --reload")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

st.divider()
st.caption("DocLens can make mistakes. Always check the cited page before relying on an answer. "
           "This is not tax advice.")

if len(st.session_state.history) > 1:
    with st.expander("Earlier questions"):
        for h in st.session_state.history[1:]:
            st.markdown(f"**{h['question']}**")
            st.write(h["answer"])