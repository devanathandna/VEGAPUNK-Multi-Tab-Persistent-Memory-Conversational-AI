import { motion } from "framer-motion";
import { Bot, User, Cpu } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessageProps {
  content: string;
  isUser: boolean;
  timestamp: string;
  model?: string;
}

export const ChatMessage = ({ content, isUser, timestamp, model }: ChatMessageProps) => {
  const renderContent = () => {
    if (isUser) {
      return <div dangerouslySetInnerHTML={{ __html: content }} className="text-sm leading-relaxed break-words" />;
    }
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
            code({node, className, children, ...props}) {
              const match = /language-(\w+)/.exec(className || '');
              // In react-markdown v10, block code has a language-* className.
              // Inline code has no className (or no language match).
              const isBlock = Boolean(match);
              return isBlock ? (
                <div className="bg-black/80 rounded-md my-2">
                  <div className="flex items-center justify-between px-4 py-1 border-b border-gray-700">
                    <span className="text-xs text-gray-400">{match![1]}</span>
                    <button
                      onClick={() => navigator.clipboard.writeText(String(children))}
                      className="text-xs text-gray-400 hover:text-white"
                    >
                      Copy
                    </button>
                  </div>
                  <pre className="p-4 overflow-x-auto">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                </div>
              ) : (
                <code className="bg-muted text-muted-foreground rounded-sm px-1" {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"} mb-4`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground flex-shrink-0">
          <Bot size={20} />
        </div>
      )}
      <div className={`max-w-[80%] ${isUser ? "text-right" : "text-left"}`}>
        <div
          className={`inline-block px-4 py-3 rounded-lg ${
            isUser
              ? "bg-user-bubble border border-terminal-green/20"
              : "bg-assistant-bubble border border-border"
          }`}
        >
          {renderContent()}
        </div>
        <div className={`text-xs text-muted-foreground mt-1 px-1 flex items-center gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
          {model && !isUser && (
            <div className="flex items-center gap-1" title={`Model: ${model}`}>
              <Cpu size={12} />
              <span>{model}</span>
            </div>
          )}
          <span>{timestamp}</span>
        </div>
      </div>
       {isUser && (
        <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-muted-foreground flex-shrink-0">
          <User size={20} />
        </div>
      )}
    </motion.div>
  );
};
