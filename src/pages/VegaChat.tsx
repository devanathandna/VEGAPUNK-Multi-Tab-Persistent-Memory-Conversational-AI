import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import { NavigationSidebar } from "@/components/NavigationSidebar";
import { MainChat } from "@/components/MainChat";
import { SubChat } from "@/components/SubChat";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  model?: string;
}

interface SubChatData {
  id: number;
  messages: Message[];
}

const VEGACHAT_API_URL = "http://localhost:5000/api";

const VegaChat = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const [mainMessages, setMainMessages] = useState<Message[]>([
    {
      id: "1",
      content: "⏳ Connecting to VEGACHAT backend...",
      isUser: false,
      timestamp: new Date().toLocaleTimeString(),
      model: "system",
    },
  ]);
  
  const [subChats, setSubChats] = useState<SubChatData[]>([]);
  const MAX_SUB_CHATS = 3;

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    let retries = 0;
    const maxRetries = 5;
    
    const attemptHealthCheck = async (): Promise<boolean> => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout instead of 3
        
        const res = await fetch(`${VEGACHAT_API_URL}/health`, {
          method: 'GET',
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const data = await res.json();
        
        if (data.ready) {
          console.log('✅ Backend is ready!');
          setBackendReady(true);
          setMainMessages([{
            id: "1",
            content: "Welcome to VEGACHAT, A Playground to Explore Multi-Conversational AI",
            isUser: false,
            timestamp: new Date().toLocaleTimeString(),
            model: "llama-3.1-8b-instant",
          }]);
          return true;
        } else {
          console.log('⏳ Backend is initializing...');
          return false;
        }
      } catch (error) {
        console.error(`❌ Health check failed (attempt ${retries + 1}/${maxRetries}):`, error);
        return false;
      }
    };
    
    // Initial check
    if (await attemptHealthCheck()) return;
    
    // Retry with exponential backoff
    const retryInterval = setInterval(async () => {
      retries++;
      if (retries >= maxRetries) {
        clearInterval(retryInterval);
        setMainMessages([{
          id: "1",
          content: "❌ Cannot connect to VEGACHAT backend. Please make sure it's running on http://localhost:5000",
          isUser: false,
          timestamp: new Date().toLocaleTimeString(),
          model: "system",
        }]);
        return;
      }
      
      if (await attemptHealthCheck()) {
        clearInterval(retryInterval);
      }
    }, 2000);
  };

  const getSubChatHeight = (index: number) => {
    const totalSubChats = subChats.length;
    switch (totalSubChats) {
      case 1:
        return "100%";
      case 2:
        return "50%";
      case 3:
        return "33.333%";
      default:
        return "100%";
    }
  };

  const addMainMessage = async (content: string, isUser: boolean = true) => {
    // Check if backend is ready
    if (!backendReady && isUser) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        content: "❌ Backend is not ready yet. Please wait...",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMainMessages((prev) => [...prev, errorMessage]);
      return;
    }
    
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMainMessages((prev) => [...prev, newMessage]);

    if (isUser) {
      const thinkingId = (Date.now() + 1).toString();
      const thinkingMessage: Message = {
        id: thinkingId,
        content: "Let me think...",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMainMessages((prev) => [...prev, thinkingMessage]);

      try {
        const res = await fetch(`${VEGACHAT_API_URL}/chat/main`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            message: content,
            session_id: sessionId 
          }),
        });

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        
        // Update session ID if it's a new session
        if (data.session_id && !sessionId) {
          setSessionId(data.session_id);
          // New session created — refresh sidebar history
          setSidebarRefreshKey((k) => k + 1);
        }
        
        const responseMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: data.response,
          isUser: false,
          timestamp: new Date().toLocaleTimeString(),
          model: data.model,
        };
        setMainMessages((prev) => prev.filter(m => m.id !== thinkingId));
        setMainMessages((prev) => [...prev, responseMessage]);
        // Refresh sidebar after every completed response so history stays current
        setSidebarRefreshKey((k) => k + 1);
      } catch (error) {
        console.error("Failed to get response from backend:", error);
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: `❌ Error: Could not connect to the VEGACHAT backend. Is it running?`,
          isUser: false,
          timestamp: new Date().toLocaleTimeString(),
        };
        setMainMessages((prev) => prev.filter(m => m.id !== thinkingId));
        setMainMessages((prev) => [...prev, errorMessage]);
      }
    }
  };

  const handleNewSubChat = () => {
    if (subChats.length < MAX_SUB_CHATS) {
      const newId = subChats.length > 0 ? Math.max(...subChats.map((sc) => sc.id)) + 1 : 1;
      setSubChats((prev) => [...prev, { id: newId, messages: [] }]);
    }
  };

  const handleCloseSubChat = (id: number) => {
    setSubChats((prev) => prev.filter((sc) => sc.id !== id));
  };

  const handleSubChatMessage = async (subChatId: number, content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
    };

    setSubChats((prev) =>
      prev.map((sc) => (sc.id === subChatId ? { ...sc, messages: [...sc.messages, newMessage] } : sc))
    );

    const thinkingId = (Date.now() + 1).toString();
    const thinkingMessage: Message = {
      id: thinkingId,
      content: "🧠 Thinking...",
      isUser: false,
      timestamp: new Date().toLocaleTimeString(),
    };
    setSubChats((prev) =>
      prev.map((sc) => (sc.id === subChatId ? { ...sc, messages: [...sc.messages, thinkingMessage] } : sc))
    );

    try {
      const res = await fetch(`${VEGACHAT_API_URL}/chat/sub`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: content, 
          subChatId,
          session_id: sessionId 
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      const responseMessage: Message = {
        id: (Date.now() + 2).toString(),
        content: data.response,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        model: data.model,
      };

      setSubChats((prev) =>
        prev.map((sc) => {
          if (sc.id === subChatId) {
            return {
              ...sc,
              messages: [...sc.messages.filter(m => m.id !== thinkingId && m.id !== newMessage.id), newMessage, responseMessage],
            };
          }
          return sc;
        })
      );
    } catch (error) {
      console.error("Failed to get response from backend:", error);
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        content: `❌ Error: Could not connect to the VEGACHAT backend.`,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };
      setSubChats((prev) =>
        prev.map((sc) => {
          if (sc.id === subChatId) {
            return {
              ...sc,
              messages: sc.messages.filter(m => m.id !== thinkingId),
            };
          }
          return sc;
        })
      );
       setSubChats((prev) =>
        prev.map((sc) => (sc.id === subChatId ? { ...sc, messages: [...sc.messages, errorMessage] } : sc))
      );
    }
  };

  const handleGroupSelect = async (groupKey: string) => {
    try {
      const encoded = encodeURIComponent(groupKey);
      const res = await fetch(`${VEGACHAT_API_URL}/sessions/group/${encoded}`);
      const data = await res.json();

      if (data.success && data.history) {
        const loadedMessages: Message[] = data.history.map((msg: any) => ({
          id: msg.id.toString(),
          content: msg.content,
          isUser: msg.role === "user",
          timestamp: new Date(msg.timestamp).toLocaleTimeString(),
        }));

        setMainMessages(loadedMessages);
        // set sessionId to first session in the group so sub-chats can continue using it
        if (data.sessions && data.sessions.length > 0) {
          setSessionId(data.sessions[0]);
        }
        // Clear sub chats when switching sessions
        setSubChats([]);
      }
    } catch (error) {
      console.error("Failed to load group session:", error);
    }
  };

  const handleNewChat = () => {
    setSessionId(null);
    setSubChats([]);
    setMainMessages([
      {
        id: "1",
        content: "Welcome to VEGACHAT. I am powered by Gemini with hybrid memory. How can I help you today?",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        model: "gemini-2.5-flash",
      },
    ]);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background fixed inset-0">
      <NavigationSidebar
        onNewSubChat={handleNewSubChat}
        onNewChat={handleNewChat}
        activeSubChats={subChats.length}
        maxSubChats={MAX_SUB_CHATS}
        onGroupSelect={handleGroupSelect}
        refreshKey={sidebarRefreshKey}
      />
      
      <div className="flex flex-1 overflow-hidden relative min-w-0">
        <MainChat
          messages={mainMessages}
          onSendMessage={addMainMessage}
          sessionId={sessionId}
        />
        
        {subChats.length > 0 && (
          <div className="w-[400px] flex flex-col border-l border-border overflow-hidden">
            <AnimatePresence>
              {subChats.map((subChat, index) => (
                <SubChat
                  key={subChat.id}
                  id={subChat.id}
                  messages={subChat.messages}
                  onSendMessage={(text) => handleSubChatMessage(subChat.id, text)}
                  onClose={() => handleCloseSubChat(subChat.id)}
                  height={getSubChatHeight(index)}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

export default VegaChat;
