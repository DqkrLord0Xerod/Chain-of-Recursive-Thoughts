// Context provider for RecThink
import React, { createContext, useContext, useState, useEffect } from 'react';
import * as api from '../api';

const RecThinkContext = createContext();

export const useRecThink = () => useContext(RecThinkContext);

export const RecThinkProvider = ({ children }) => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingProcess, setThinkingProcess] = useState(null);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('mistralai/mistral-small-3.1-24b-instruct:free');
  const [models, setModels] = useState([]);
  const [thinkingRounds, setThinkingRounds] = useState('auto');
  const [alternativesPerRound, setAlternativesPerRound] = useState(3);
  const [budgetCap, setBudgetCap] = useState(100000);
  const [enforceBudget, setEnforceBudget] = useState(true);
  const [error, setError] = useState(null);
  const [showThinkingProcess, setShowThinkingProcess] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [websocket, setWebsocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Initialize chat session
  const initializeChat = async () => {
    try {
      setError(null);
      const result = await api.initializeChat(
        apiKey,
        model,
        budgetCap,
        enforceBudget
      );
      setSessionId(result.session_id);
      
      // Initialize with welcome message
      setMessages([
        { role: 'assistant', content: 'Welcome to RecThink! I use recursive thinking to provide better responses. Ask me anything.' }
      ]);
      
      // Set up WebSocket connection
      const ws = api.createWebSocketConnection(result.session_id);
      setWebsocket(ws);
      
      return result.session_id;
    } catch (err) {
      setError(err.message);
      return null;
    }
  };

  // Send a message and get response
  const sendMessage = async (content) => {
    try {
      setError(null);
      
      // Add user message to conversation
      const newMessages = [...messages, { role: 'user', content }];
      setMessages(newMessages);
      
      // Start thinking process
      setIsThinking(true);
      
      // Determine thinking rounds if set to auto
      let rounds = null;
      if (thinkingRounds !== 'auto') {
        rounds = parseInt(thinkingRounds, 10);
      }
      
      // Send message via API
      const result = await api.sendMessage(sessionId, content, {
        thinkingRounds: rounds,
        alternativesPerRound
      });

      // Update conversation with assistant's response
      setMessages([...newMessages, { role: 'assistant', content: result.response }]);

      // Store thinking process
      setThinkingProcess({
        rounds: result.thinking_rounds,
        history: result.thinking_history
      });

      const costData = await api.getCost(sessionId);
      setCost({
        tokenLimit: costData.token_limit,
        tokensUsed: costData.tokens_used,
        dollarsSpent: costData.dollars_spent,
      });

      setIsThinking(false);
      return result;
    } catch (err) {
      setError(err.message);
      setIsThinking(false);
      return null;
    }
  };

  // Save conversation
  const saveConversation = async (filename = null, fullLog = false) => {
    try {
      setError(null);
      return await api.saveConversation(sessionId, filename, fullLog);
    } catch (err) {
      setError(err.message);
      return null;
    }
  };

  // Load sessions
  const loadSessions = async () => {
    try {
      setError(null);
      const result = await api.listSessions();
      setSessions(result.sessions);
      return result.sessions;
    } catch (err) {
      setError(err.message);
      return [];
    }
  };

  // Delete session
  const deleteSession = async (id) => {
    try {
      setError(null);
      await api.deleteSession(id);
      
      // Update sessions list
      const updatedSessions = sessions.filter(session => session.session_id !== id);
      setSessions(updatedSessions);
      
      // Clear current session if it's the one being deleted
      if (id === sessionId) {
        setSessionId(null);
        setMessages([]);
        setThinkingProcess(null);
      }
      
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    }
  };

  const finalizeSession = async () => {
    if (!sessionId) return null;
    try {
      setError(null);
      const result = await api.finalizeSession(sessionId);
      setSessionId(null);
      setMessages([...messages, { role: 'assistant', content: result.summary }]);
      setThinkingProcess(null);
      setCost({
        tokenLimit: result.cost.token_limit,
        tokensUsed: result.cost.tokens_used,
        dollarsSpent: result.cost.dollars_spent,
      });
      return result.summary;
    } catch (err) {
      setError(err.message);
      return null;
    }
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const result = await api.getModels();
        setModels(result.models);
      } catch (err) {
        // Ignore fetching errors in UI context
      }
    };
    loadModels();
  }, []);

  // Set up WebSocket listeners when connection is established
  useEffect(() => {
    if (!websocket) return;
    
    websocket.onopen = () => {
      setConnectionStatus('connected');
    };
    
    websocket.onclose = () => {
      setConnectionStatus('disconnected');
    };
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'chunk') {
        // Handle streaming chunks for real-time updates during thinking
        // This could update a temporary display of thinking process
      } else if (data.type === 'final') {
        // Handle final response with complete thinking history
        setMessages(prev => [...prev.slice(0, -1), { role: 'assistant', content: data.response }]);
        setThinkingProcess({
          rounds: data.thinking_rounds,
          history: data.thinking_history
        });
        setCost(prev => ({
          ...prev,
          dollarsSpent: data.cost_total,
        }));
        setIsThinking(false);
      } else if (data.error) {
        setError(data.error);
        setIsThinking(false);
      }
    };
    
    websocket.onerror = (error) => {
      setError('WebSocket error: ' + error.message);
      setConnectionStatus('error');
    };
    
    // Clean up function
    return () => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, [websocket]);

  // Periodically fetch cost metrics
  useEffect(() => {
    if (!sessionId) return undefined;

    const interval = setInterval(async () => {
      try {
        const data = await api.getCost(sessionId);
        setCost({
          tokenLimit: data.token_limit,
          tokensUsed: data.tokens_used,
          dollarsSpent: data.dollars_spent,
        });
      } catch (e) {
        // ignore errors in polling
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [sessionId]);

  // Context value
  const value = {
    sessionId,
    messages,
    isThinking,
    thinkingProcess,
    apiKey,
    model,
    models,
    thinkingRounds,
    alternativesPerRound,
    budgetCap,
    enforceBudget,
    error,
    showThinkingProcess,
    sessions,
    connectionStatus,
    cost,
    
    // Setters
    setApiKey,
    setModel,
    setThinkingRounds,
    setAlternativesPerRound,
    setShowThinkingProcess,
    setBudgetCap,
    setEnforceBudget,
    
    // Actions
    initializeChat,
    sendMessage,
    saveConversation,
    loadSessions,
    deleteSession,
    finalizeSession
  };

  return (
    <RecThinkContext.Provider value={value}>
      {children}
    </RecThinkContext.Provider>
  );
};

export default RecThinkContext;
