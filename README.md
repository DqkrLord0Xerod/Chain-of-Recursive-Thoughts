# CoRT (Chain of Recursive Thoughts) 🧠🔄

## How it works
1. AI generates initial response
2. AI decides how many "thinking rounds" it needs
3. For each round:
   - Generates 3 alternative responses
   - Evaluates all responses
   - Assigns an overlap score to each option
   - Combines that score with the model's pick to choose the winner
4. Final response is the survivor of this AI battle royale

## How to use the Web UI(still early dev)
1. Open start_recthink.bat
2. wait for a bit as it installs dependencies
3. profit??

If running on Linux:
```
pip install -r requirements.txt
cd frontend && npm install
cd ..
python recthink_web.py  # start the API server
```

(open a new shell)

```
cd frontend
npm start
```

## Try it yourself
### CLI usage
```bash
pip install -r requirements.txt
# Option 1: export your API key
export OPENROUTER_API_KEY="your-key-here"
# Option 2: place it in a `.env` file
# OPENROUTER_API_KEY=your-key-here
# Select environment (development|staging|production)
export APP_ENV=development
# Optional overrides
# API_BASE_URL=https://api.example.com
# WS_BASE_URL=wss://api.example.com
python -m cli.main
```
You can also limit the context window by setting `max_context_tokens` in a
`CoRTConfig` instance when creating `EnhancedRecursiveThinkingChat`.

### The Secret Sauce
The magic is in:

 - Self-evaluation
 - Competitive alternative generation
 - Iterative refinement
 - Dynamic thinking depth


### Running tests
Install the requirements and execute pytest:
```bash
pip install -r requirements.txt
pip install pytest flake8
pytest
```

To run only the integration scenarios:
```bash
pytest tests/test_integration.py
```

### License
MIT - Go wild with it
