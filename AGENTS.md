# Repository Guidelines

## Code Style
- Follow **PEP-8** for all Python code.
- Keep lines under 88 characters when possible.
- Use descriptive variable and function names.

## Linting
- Run `flake8` on modified Python files before committing.
- Address any warnings or errors raised by the linter.

## Commit Messages
- Use short messages following the pattern `type: summary`.
- Common types: `feat`, `fix`, `docs`, `chore`, `test`.
- Example: `feat: add streaming websocket support`.

## Running the Project
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  cd frontend && npm install
  ```
- Set `OPENROUTER_API_KEY` in your environment before running:
  ```bash
  export OPENROUTER_API_KEY="your-key-here"
  ```
- Command line usage:
  ```bash
  python recursive-thinking-ai.py
  ```
- Web application:
  ```bash
  python recthink_web.py       # backend API
  # In a separate shell
  cd frontend && npm start     # frontend UI
  ```

## Tests
- Place Python tests under a `tests/` directory and run them with `pytest`.
- Frontend tests run with `npm test`.
- Always run all tests before opening a pull request.
