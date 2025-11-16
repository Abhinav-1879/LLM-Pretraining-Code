import os, sys
from langchain_groq import ChatGroq

key = os.getenv("GROQ_API_KEY")
if not key:
    print("Set GROQ_API_KEY env var or hardcode it in this file"); sys.exit(1)

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=key)
prompt = "Explain LLM pretraining in one sentence."

try:
    out = model(prompt)
    print(out)
except Exception:
    try:
        from langchain.schema import HumanMessage
    except Exception:
        try:
            from langchain_core.schema import HumanMessage
        except Exception:
            HumanMessage = None

    if HumanMessage:
        msgs = [HumanMessage(content=prompt)]
        try:
            res = model.generate([msgs])
            gens = getattr(res, "generations", None) or getattr(res, "generation", None)
            if gens:
                first = gens[0]
                if isinstance(first, list):
                    print(first[0].text)
                else:
                    print(first.text if hasattr(first, "text") else first)
            else:
                print(res)
        except Exception as e:
            print("chat generate failed:", e)
    else:
        print("Both simple and chat-style calls failed.")
