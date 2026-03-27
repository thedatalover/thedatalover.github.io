# Building a Dual-Context AI Agent with Elasticsearch Managed Memory

What if your AI agent could remember only what's needed for each user—and forget everything else? In this hands-on tutorial, you'll build exactly that: an agent with selective, context-aware memory powered by Elasticsearch. By the end, you'll have a working system where memory isn't just stored, it's controlled.

## The core problem: Why LLMs don't have memory of their own?

Before writing a single line of code, it's worth understanding what we're actually solving. Here's something that surprises many developers: conversations with LLMs are completely [stateless](https://arxiv.org/abs/2603.07670). Every time you send a message, you must include the entire conversation history to "remind" the model what happened before. The ability to maintain continuity within a single session is what we call **short-term memory**.

Long-term memory is a step further. When we want to persist information — like user preferences — across entirely separate conversations, we inject that information into new sessions as needed. The model never truly "remembers" anything; we just make sure the right context is always in the room.

If we're already managing the context, why stop at appending messages? Here are three reasons to go further.

First, we can inject useful context—slipping in relevant facts from past interactions. Second, we can summarize and prune information that's no longer needed to avoid context poisoning. Third, we save tokens and keep the context window efficient for longer, more focused conversations.

## The mental model: Neo's dual identity

To put it simply, think about Neo from *The Matrix*. He exists simultaneously as Thomas A. Anderson — an ordinary software developer living inside the simulation — and as Neo, a liberated operative working with the resistance in the real world of Zion. The moment he plugs in or unplugs from the Matrix, his entire operational context switches. Information from one world has no business leaking into the other.

That's exactly the architecture we're building. Our agent, Neo, will maintain two completely isolated memory pools: **Matrix memories** for in-simulation interactions, and **Zion memories** for real-world operations. [Elasticsearch's document-level security](https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/controlling-access-at-document-field-level) will enforce that boundary automatically without any manual filtering required.

## What types of memory does our agent need?

Not all memories serve the same purpose, and a flat list of chat messages will only take you so far. Modern agent architectures — including the [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427) framework — distinguish between three types of memory, each requiring distinct storage and retrieval strategies. Let's walk through each one.

* **Procedural memory** defines *how* the agent behaves, not what it knows. Think of it as Neo's uploaded combat training — the kung fu, the tactics, the rules of engagement. It governs when to store a memory, when to retrieve one, how to summarize conversations, and how to use tools. In our system, procedural memory lives in the application code and prompt instructions. It *uses* Elasticsearch rather than being stored in it.

* **Episodic memory** captures specific experiences tied to a person and a moment in time. For example: "Trinity told Neo the agents are watching the downtown exit" or "Morpheus has a meeting with the Oracle at 9 am." This is the most personal and dynamic form of memory, and the most dangerous to get wrong — a leak between contexts here is exactly the kind of thing that gets operatives killed (or, less dramatically, makes your chatbot embarrassingly confused). Each episodic memory in our system is stored as an Elasticsearch document, with metadata capturing the user, timestamp, and context type (Matrix or Zion).

* **Semantic memory** is shared world knowledge — facts that are true regardless of who's asking or when. In our analogy, this is Neo's understanding of the Machines, the structure of Zion, and how the simulation works. It doesn't belong to any one conversation; it's the backdrop against which everything else is reasoned. Documents like operational manuals for the [Nebuchadnezzar](https://en.wikipedia.org/wiki/Nebuchadnezzar_(The_Matrix)) serve this role. Unlike episodic retrieval (which needs tight filters), semantic retrieval favors broad, concept-level search designed to surface generally true information.

With these memory types in mind, we're ready to build the system.

## Prerequisites

To follow along, you'll need an Elasticsearch Elastic Cloud Hosted (ECH) or self-hosted 9.1+ instance, Python 3.x, and an [OpenAI API key](https://platform.openai.com/docs/api-reference/authentication). Start by installing the required packages:

```bash
pip install openai elasticsearch==9.1.0 python-dotenv
```

## Store your credentials in a `.env` file

To avoid hardcoding secrets into our script or typing them interactively each run, we'll use a `.env` file to manage all connection settings in one place. Create a file named `.env` in the root of your project directory with the following contents:

```ini
OPENAI_API_KEY=your_openai_api_key_here
ELASTICSEARCH_URL=https://your-cluster.es.io:9243
ELASTICSEARCH_API_KEY=your_elasticsearch_api_key_here
```

> **Important:** Add `.env` to your `.gitignore` file immediately. This single habit prevents credentials from being committed to version control.

## Step 1 — Connect to OpenAI and Elasticsearch

With the `.env` file in place, we can now load those values at runtime using the `python-dotenv` library. Think of `load_dotenv()` as the step that reads your `.env` file and injects its contents into the process's environment variables, making them available to `os.getenv()` throughout the rest of the script.

```python
from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

# Load all variables from the .env file into the environment.
# This must be called before any os.getenv() calls.
load_dotenv()

# Initialize the OpenAI client using the key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pull Elasticsearch connection settings from the environment
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
ELASTICSEARCH_INDEX = "memories"

# Admin client — used for index/role/user management
es_client = Elasticsearch(
    hosts=[ELASTICSEARCH_URL],
    api_key=ELASTICSEARCH_API_KEY
)

# Quick connectivity check — if this prints cluster info, you're connected
print(es_client.info())
```

## Step 2 — Design the memory index

The schema below is the backbone of everything. Notice that `memory_text` is defined as a [multi-field](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/multi-fields): it stores both a plain-text version (for keyword search) and a `semantic_text` sub-field (for vector-based retrieval using the [Elastic Learned Sparse EncodeR (ELSER) model](https://www.elastic.co/search-labs/tutorials/search-tutorial/semantic-search/elser-model)). This gives us [semantic search](https://www.elastic.co/search-labs/blog/introduction-to-vector-search) over the same content — precise when we need it, conceptual when we don't.

```python
from datetime import datetime

mappings = {
    "properties": {
        "user_id":      {"type": "keyword"},
        "memory_type":  {"type": "keyword"},   # "matrix" or "zion"
        "created_at":   {"type": "date"},
        "memory_text": {
            "type": "text",
            "fields": {
                # The semantic sub-field enables vector search via ELSER
                "semantic": {"type": "semantic_text"}
            }
        }
    }
}

try:
    es_client.indices.create(
        index=ELASTICSEARCH_INDEX,
        mappings=mappings,
        ignore=400  # Ignore "already exists" errors on re-runs
    )
    print(f"Index '{ELASTICSEARCH_INDEX}' created successfully.")
except Exception as e:
    print(f"Error creating index: {e}")
```

## Step 3 — Seed some initial memories

Let's populate the index with a few memories to test against. Notice that each document declares its `memory_type` — this is the field that document-level security will use to enforce context isolation.

```python
memories = [
    {
        "user_id":      "trinity99",
        "memory_type":  "zion",           # Visible only to Zion-side users
        "created_at":   datetime.now(),
        "memory_text":  "Trinity and Neo agreed: if they get separated, "
                        "the emergency extraction point is the Adams Street phone booth."
    },
    {
        "user_id":      "switch_operator",
        "memory_type":  "matrix",         # Visible only to Matrix-side users
        "created_at":   datetime.now(),
        "memory_text":  "The target agent always uses the Wachowski Building "
                        "entrance at 9am sharp."
    },
]

# Bulk index for efficiency
operations = []
for mem in memories:
    operations.append({"index": {"_index": ELASTICSEARCH_INDEX}})
    operations.append(mem)

try:
    response = es_client.bulk(operations=operations)
    print(f"Indexed {len(memories)} memories successfully.")
except Exception as e:
    print(f"Bulk indexing error: {e}")
```

> **Note:** The first run may time out briefly while the ML nodes warm up the ELSER model. Wait a minute and retry if that happens.

## Step 4 — Create roles with built-in security filters

This is where the architecture gets elegant. Rather than writing security logic into our application, we push it down to the database layer. We define two Elasticsearch roles — one for each context — each with a document-level query filter baked in. Any user carrying the `matrix` role will *only ever see* documents where `memory_type` equals `"matrix"`, no matter what query they run.

```python
# The Matrix-side role: can only read/write simulation memories
matrix_role = {
    "indices": [{
        "names":      ["memories"],
        "privileges": ["read", "write"],
        "query": {
            "bool": {
                "filter": [{"term": {"memory_type": "matrix"}}]
            }
        }
    }]
}

# The Zion-side role: can only read/write real-world memories
zion_role = {
    "indices": [{
        "names":      ["memories"],
        "privileges": ["read", "write"],
        "query": {
            "bool": {
                "filter": [{"term": {"memory_type": "zion"}}]
            }
        }
    }]
}

try:
    es_client.security.put_role(name="matrix", body=matrix_role)
    print("Role 'matrix' created.")
    es_client.security.put_role(name="zion", body=zion_role)
    print("Role 'zion' created.")
except Exception as e:
    print(f"Error creating roles: {e}")
```

You can explore more examples of access control [here](https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/controlling-access-at-document-field-level#basic-examples) and learn more about role management [here](https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/kibana-role-management).

## Step 5 — Create users and assign them to roles

Now we create the actual users. Trinity operates on the Zion side; Switch operates inside the Matrix. Each user gets credentials tied to their role, so Elasticsearch automatically determines what they can see.

```python
# Trinity: a Zion-side operative — sees only real-world memories
trinity_user = {
    "password": "R3dP1ll$ecure!",
    "roles":    ["zion"],
    "full_name": "Trinity",
    "email":    "trinity99@zion.net"
}

# Switch: a Matrix-side operative — sees only simulation memories
switch_user = {
    "password": "Blu3P1ll$ecure!",
    "roles":    ["matrix"],
    "full_name": "Switch",
    "email":    "switch@matrix.sim"
}

try:
    es_client.security.put_user(username="trinity99",       body=trinity_user)
    es_client.security.put_user(username="switch_operator", body=switch_user)
    print("Users created successfully.")
except Exception as e:
    print(f"Error creating users: {e}")
```

## Step 6 — Verify isolation, since we want to keep Zion safe

Before building the agent, it's worth proving the isolation actually works. Let's query the index for each user and confirm they only see their own context's memories.

```python
# Query as Trinity (Zion role) — should only return "zion" memories
trinity_client = Elasticsearch(
    hosts=[ELASTICSEARCH_URL],
    basic_auth=("trinity99", "R3dP1ll$ecure!")
)
trinity_results = trinity_client.search(
    index=ELASTICSEARCH_INDEX,
    query={"match_all": {}}
)
print("Trinity sees:", [h["_source"]["memory_type"]
                        for h in trinity_results["hits"]["hits"]])
# Expected output: ["zion"]

# Query as Switch (Matrix role) — should only return "matrix" memories
switch_client = Elasticsearch(
    hosts=[ELASTICSEARCH_URL],
    basic_auth=("switch_operator", "Blu3P1ll$ecure!")
)
switch_results = switch_client.search(
    index=ELASTICSEARCH_INDEX,
    query={"match_all": {}}
)
print("Switch sees:", [h["_source"]["memory_type"]
                       for h in switch_results["hits"]["hits"]])
# Expected output: ["matrix"]
```

If the output matches the expected values, the security layer is working perfectly. The same index, the same query — two completely different views of the data.

## Step 7 — Define the agent's tools

Our agent, Neo, will use three tools to reason. `GetKnowledge` handles [RAG-style retrieval](https://www.elastic.co/search-labs/blog/retrieval-augmented-generation-rag) from a static knowledge base. `GetMemories` fetches relevant episodic memories using [hybrid search](https://www.elastic.co/what-is/hybrid-search). `SetMemory` persists new information from the conversation. The LLM will decide autonomously which tools to call and when to call them.

```python
tools = [
    {
        "type": "function",
        "name": "GetKnowledge",
        "description": "Search the agent's internal knowledge base for relevant context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search the knowledge base."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "GetMemories",
        "description": "Retrieve memories from past conversations relevant to the current question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search the memory store."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "SetMemory",
        "description": "Save a new memory if the current message contains something worth remembering.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The information to store as a memory."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
]
```

## Step 8 — Implement the tool functions

Now we wire up the actual logic behind those tool definitions. The most important thing to notice in `get_memory` is what means *absent*: there are no manual security filters in the query. Elasticsearch automatically enforces access control based on the client's credentials.

```python
import json

def get_knowledge(query: str) -> str:
    # Placeholder — in production, this would query a separate knowledge index
    return "Empty knowledge base."


def get_memory(query: str, username: str, password: str) -> str:
    """
    Retrieves memories using hybrid search (semantic + keyword via RRF ranking).
    Security filtering is handled entirely by Elasticsearch based on user credentials —
    no application-level filtering needed here.
    """
    user_client = Elasticsearch(
        hosts=[ELASTICSEARCH_URL],
        basic_auth=(username, password)
    )

    es_query = {
        "retriever": {
            "rrf": {
                # RRF (Reciprocal Rank Fusion) blends two retrieval strategies:
                "retrievers": [
                    {
                        # 1. Semantic retrieval: finds conceptually similar memories
                        "standard": {
                            "query": {
                                "semantic": {
                                    "field":  "memory_text.semantic",
                                    "query":  query
                                }
                            }
                        }
                    },
                    {
                        # 2. Keyword retrieval: finds exact or near-exact matches
                        "standard": {
                            "query": {
                                "multi_match": {
                                    "query":  query,
                                    "fields": ["memory_text"]
                                }
                            }
                        }
                    }
                ],
                "rank_window_size": 50,
                "rank_constant":    20
            }
        }
    }

    response = user_client.search(index=ELASTICSEARCH_INDEX, body=es_query)

    # Format the results for the LLM
    result = "Memories\n"
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        result += f"{src['user_id']}: ({src['memory_text']})\n"

    return result


def set_memory(query: str) -> str:
    # Placeholder — in production, this would use an LLM to extract and store
    # a structured memory record from the raw conversation text
    return f"Memory saved: {query}"


def build_tool_response(call_id: str, result: str) -> dict:
    """Helper to format a tool result back into the message history."""
    return {
        "type":    "function_call_output",
        "call_id": call_id,
        "output":  str(result)
    }
```

## Step 9 — Build the agent loop

Here's where everything comes together. The agent loop follows a simple two-pass pattern: first, call the LLM with the tools available (it decides what to call), then execute those tools and call the LLM again with the results so it can generate a final answer. The critical parameter here is `username` — it determines which Elasticsearch credentials are used, and therefore which memories are visible.

```python
def run_agent(question: str, username: str, password: str) -> str:
    """
    Runs a single turn of the Neo agent.

    The `username` and `password` arguments determine which Elasticsearch
    user is active — and therefore which memory context is visible.
    Swapping these is all it takes to switch Neo's entire memory world.
    """
    messages = [
        {
            "role":    "system",
            "content": (
                "You are Neo, an intelligent agent. Always call GetKnowledge "
                "and GetMemories once before answering to gather relevant context. "
                "If the user shares something worth remembering, call SetMemory."
            )
        },
        {"role": "user", "content": question}
    ]

    # --- Pass 1: Let the LLM decide which tools to call ---
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=messages,
        tools=tools,
        parallel_tool_calls=True  # GetKnowledge and GetMemories can run simultaneously
    )

    # --- Execute each tool the LLM requested ---
    for tool_call in response.output:
        if getattr(tool_call, "type", None) != "function_call":
            continue  # Skip non-tool output blocks (e.g. text)

        name    = tool_call.name
        call_id = tool_call.call_id
        args    = json.loads(getattr(tool_call, "arguments", "{}"))
        query   = args.get("query", "")

        if name == "GetMemories":
            # Pass user credentials so Elasticsearch enforces the right context
            result = get_memory(query, username, password)
        elif name == "GetKnowledge":
            result = get_knowledge(query)
        elif name == "SetMemory":
            result = set_memory(query)
        else:
            result = f"Unknown tool: {name}"

        print(f"Tool called: {name} → {result}")

        # Append the tool result to the conversation so the LLM can use it
        messages.append({
            "role":    "assistant",
            "content": [{"type": "output_text", "text": json.dumps(
                build_tool_response(call_id, result)
            )}]
        })

    # --- Pass 2: Generate the final answer with tool results in context ---
    final_response = client.responses.create(
        model="gpt-4.1-mini",
        input=messages
    )

    return final_response.output[0].content[0].text
```

## Step 10 — Test selective memory in action

Time to run the agent and verify that the memory isolation actually holds. We'll ask both users the same question and confirm that Neo's answer changes based on who's asking.

```python
# --- Zion-side conversation (Trinity) ---
print("=== Talking to Neo as Trinity (Zion context) ===\n")

answer = run_agent(
    question="Where do we meet if things go wrong on the mission?",
    username="trinity99",
    password="R3dP1ll$ecure!"
)
print(f"Neo: {answer}\n")
# Expected: Neo recalls the Adams Street phone booth extraction point

answer = run_agent(
    question="What time does the target enter the Wachowski Building?",
    username="trinity99",
    password="R3dP1ll$ecure!"
)
print(f"Neo: {answer}\n")
# Expected: Neo has no information — that's a Matrix-side memory, invisible here


# --- Matrix-side conversation (Switch) ---
print("=== Talking to Neo as Switch (Matrix context) ===\n")

answer = run_agent(
    question="What do we know about the target's daily routine?",
    username="switch_operator",
    password="Blu3P1ll$ecure!"
)
print(f"Neo: {answer}\n")
# Expected: Neo recalls the 9am Wachowski Building entrance pattern

answer = run_agent(
    question="What's the emergency extraction point?",
    username="switch_operator",
    password="Blu3P1ll$ecure!"
)
print(f"Neo: {answer}\n")
# Expected: Neo has no information — that's a Zion-side memory, invisible here
```

If everything is working correctly, Neo should answer the first question for Trinity and draw a blank on the second — and answer the first question for Switch while drawing a blank on Trinity's extraction point. Same agent, same index, completely isolated experiences. Just like being plugged in or unplugged from the Matrix.

## Step 11 — Clean up: delete the index, users, and roles

Once you're done experimenting, it's good practice to tear down everything the tutorial created. More importantly, if you want to **re-run the tutorial from scratch**, running this cleanup block ensures you won't hit "already exists" errors on the index, roles, or users — which would otherwise interrupt the setup steps.

Think of this as the mirror image of Steps 2 through 5: for every resource we created (index → roles → users), we delete in reverse order (users → roles → index). The reverse order matters because in a production system, you'd want to remove access before removing data, reducing the window where a user could theoretically still query a resource you're in the process of deleting.

```python
# --- 1. Delete users ---
# Removing users first revokes their access credentials immediately,
# before we touch the roles or index they depended on.
for username in ["trinity99", "switch_operator"]:
    try:
        es_client.security.delete_user(username=username)
        print(f"User '{username}' deleted.")
    except Exception as e:
        # It's safe to ignore "not found" errors on re-runs where
        # the user was already deleted or never created.
        print(f"Could not delete user '{username}': {e}")

# --- 2. Delete roles ---
# With no users assigned to these roles, deleting them is now safe.
for role in ["matrix", "zion"]:
    try:
        es_client.security.delete_role(name=role)
        print(f"Role '{role}' deleted.")
    except Exception as e:
        print(f"Could not delete role '{role}': {e}")

# --- 3. Delete the memories index ---
# This removes all stored memory documents along with the index mappings.
# On the next run, Step 2 will recreate it cleanly from scratch.
try:
    es_client.indices.delete(index=ELASTICSEARCH_INDEX)
    print(f"Index '{ELASTICSEARCH_INDEX}' deleted.")
except Exception as e:
    print(f"Could not delete index '{ELASTICSEARCH_INDEX}': {e}")
```

After running this block, your Elasticsearch cluster is back to the state it was in before you started, and the tutorial is ready to be run again from Step 1 onward.

## How it all fits together

Let's step back and look at the full picture. Procedural memory (the system prompt and application logic) governs *when* Neo searches his memories and what he does with the results. Episodic memory (the documents in Elasticsearch, filtered by role) gives Neo *personal, context-specific* knowledge tied to individual operatives. Semantic memory (a knowledge index, not built here but plugged in via `GetKnowledge`) provides *shared world knowledge* that transcends any single context.

Selective retrieval is the thread that ties it together. By narrowing the search space with structured filters *before* running semantic retrieval, Elasticsearch scores fewer vectors, the LLM receives a smaller and cleaner context window, and the result is lower latency, lower token usage, and more focused reasoning — all at the same time.

Elasticsearch makes this possible through its combination of [hybrid search](https://www.elastic.co/docs/solutions/search/hybrid-semantic-text), rich metadata support, [document-level security](https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/controlling-access-at-document-field-level), and temporal filtering. The agent's "brain" is genuinely split between worlds. The difference from science fiction is that here, the split is intentional, auditable, and useful — not a glitch in the simulation.