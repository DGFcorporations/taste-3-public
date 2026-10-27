# taste-3

This repository contains the public mechanics of TASTE variant 3, a temporal AI standards engine. It solves the problem of allowing AI agents and systems to evaluate context against a personal set of rules, standards, and directives, without hardcoding private behavioral data into the agents themselves. By separating the engine from the private data, the system can enforce persona-driven behavior and semantic rule retrieval while keeping the owner's knowledge base secure.

### Tech Stack
- **Language:** Python (3.11+)
- **Core Frameworks:** FastAPI, Pydantic
- **AI / Semantic:** FastEmbed (local embedding models)
- **Protocols & Serving:** MCP (Model Context Protocol), Granian (ASGI server)

### How to Run

Install the engine and its dependencies (ideally in a virtual environment):
```bash
pip install -e .
```

The CLI provides commands for managing the standards graph and running the servers:
```bash
# Run the high-performance Granian HTTP server
taste serve --socket ./data/taste.sock

# Run the MCP (Model Context Protocol) stdio server
taste mcp

# Use operator commands to manage the temporal graph (e.g., export, import, drift)
taste --help
```

### Key Features
- **Semantic Retrieval:** Uses `FastEmbed` to find relevant behavioral standards based on contextual queries.
- **Temporal Graph:** Standards have a validity window and supersession history, enabling point-in-time drift reporting and safe evolution of rules.
- **Profile Isolation:** Supports multiple personas/profiles operating in isolated graphs.
- **Multi-Protocol:** Exposes logic via HTTP (FastAPI) and Model Context Protocol (MCP) for agent integration.

### What's Never Here
The owner's corpus (standards, decision records, rationale, persona-derived layers) lives in a private repository and represents the system's learned knowledge — publishing them would publish the person.

Each trial cycle: corpus pruned → keepers stay private → mechanics worth sharing land here via the publish decision.
