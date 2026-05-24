import { motion } from "framer-motion";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { useEffect, useRef } from "react";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  model?: string;
}

interface MainChatProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  sessionId: string | null;
}

export const MainChat = ({ messages, onSendMessage, sessionId }: MainChatProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 h-screen flex flex-col bg-background">
      {/* Header */}
      <div className="px-6 py-4 border-b border-terminal-green/30 bg-card/50 backdrop-blur-sm">
        <h2 className="text-lg font-semibold text-terminal-green flex items-center gap-2">
          <span className="text-terminal-green">$</span> VegaCHAT
          {sessionId && <span className="text-xs text-muted-foreground font-normal">• Session Active</span>}
        </h2>
        <p className="text-xs text-muted-foreground mt-1">Advanced AI chat system with hybrid memory and parallel thinking.</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-2">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-2">
              <p className="text-terminal-green text-lg font-semibold">$ ready_</p>
              <p className="text-muted-foreground text-sm">Start a conversation...</p>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <ChatMessage
              key={msg.id}
              content={msg.content}
              isUser={msg.isUser}
              timestamp={msg.timestamp}
              model={msg.model}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={onSendMessage} placeholder="$ type your message..." isMainChat={true} />
    </div>
  );
};
