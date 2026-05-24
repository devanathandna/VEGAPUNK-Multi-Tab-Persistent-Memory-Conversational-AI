import { motion } from "framer-motion";
import { X, Plus } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { Button } from "./ui/button";
import { useEffect, useRef } from "react";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: string;
  model?: string;
}

interface SubChatProps {
  id: number;
  messages: Message[];
  onSendMessage: (message: string) => void;
  onClose: () => void;
  height: string;
}

export const SubChat = ({ id, messages, onSendMessage, onClose, height }: SubChatProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1, height }}
      exit={{ y: 20, opacity: 0 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="flex flex-col bg-background border-b border-border"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-terminal-green/30 bg-card/50 backdrop-blur-sm flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-terminal-green flex items-center gap-2">
            <span className="text-terminal-green">$</span> Satellite #{id}
          </h3>
          <p className="text-xs text-muted-foreground">Parallel thought stream</p>
        </div>
        <div className="flex gap-1">
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 text-terminal-green hover:bg-terminal-green/10"
            title="Append to main chat"
          >
            <Plus className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 text-muted-foreground hover:text-foreground hover:bg-card"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-2">
              <p className="text-terminal-green text-sm font-semibold">$ satellite_{id}_</p>
              <p className="text-muted-foreground text-xs">New thought stream</p>
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
      <ChatInput onSend={onSendMessage} placeholder={`$ satellite #${id}...`} />
    </motion.div>
  );
};
