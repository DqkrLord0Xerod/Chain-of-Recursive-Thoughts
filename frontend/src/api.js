// API client for RecThink
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

export const initializeChat = async (
  apiKey,
  model,
  budgetTokenLimit,
  enforceBudget = true
) => {
  const response = await fetch(`${API_BASE_URL}/initialize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      api_key: apiKey,
      model,
      budget_token_limit: budgetTokenLimit,
      enforce_budget: enforceBudget,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to initialize chat: ${response.statusText}`);
  }
  
  return response.json();
};

export const sendMessage = async (sessionId, message, options = {}) => {
  const response = await fetch(`${API_BASE_URL}/send_message`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      thinking_rounds: options.thinkingRounds,
      alternatives_per_round: options.alternativesPerRound,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to send message: ${response.statusText}`);
  }
  
  return response.json();
};

export const saveConversation = async (sessionId, filename = null, fullLog = false) => {
  const response = await fetch(`${API_BASE_URL}/save`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      filename,
      full_log: fullLog,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to save conversation: ${response.statusText}`);
  }
  
  return response.json();
};

export const listSessions = async () => {
  const response = await fetch(`${API_BASE_URL}/sessions`);
  
  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.statusText}`);
  }
  
  return response.json();
};

export const getCost = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/cost/${sessionId}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch cost: ${response.statusText}`);
  }

  return response.json();
};

export const finalizeSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/finalize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!response.ok) {
    throw new Error(`Failed to finalize session: ${response.statusText}`);
  }

  return response.json();
};

export const deleteSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
  
  return response.json();
};

export const getModels = async () => {
  const response = await fetch(`${API_BASE_URL}/models`);

  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }

  return response.json();
};

export const createWebSocketConnection = (sessionId) => {
  const wsBase = process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000';
  const ws = new WebSocket(`${wsBase}/ws/${sessionId}`);
  return ws;
};
