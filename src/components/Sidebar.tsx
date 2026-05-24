import { motion } from "framer-motion";
import { Plus, Terminal, MessageSquare } from "lucide-react";
import { Button } from "./ui/button";

interface SidebarProps {
  onNewSubChat: () => void;
  activeSubChats: number;
  maxSubChats: number;
}

export const Sidebar = ({ onNewSubChat, activeSubChats, maxSubChats }: SidebarProps) => {
  return (
    <motion.div
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-64 h-screen bg-card border-r border-border flex flex-col p-4"
    >
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <Terminal className="h-5 w-5 text-terminal-green" />
          <h1 className="text-lg font-bold text-terminal-green">VEGAPUNK</h1>
        </div>
        <p className="text-xs text-muted-foreground">Parallel Mind System</p>
      </div>

      {/* New Satellite Button */}
      <Button
        onClick={onNewSubChat}
        disabled={activeSubChats >= maxSubChats}
        className="w-full mb-4 bg-terminal-green hover:bg-terminal-glow text-primary-foreground terminal-glow"
      >
        <Plus className="h-4 w-4 mr-2" />
        New Satellite
      </Button>

      {/* Status */}
      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between p-2 rounded bg-background border border-border">
          <span className="text-muted-foreground">Active Satellites:</span>
          <span className="text-terminal-green font-semibold">{activeSubChats}</span>
        </div>
        <div className="flex items-center justify-between p-2 rounded bg-background border border-border">
          <span className="text-muted-foreground">Max Satellites:</span>
          <span className="text-foreground font-semibold">{maxSubChats}</span>
        </div>
      </div>

      {/* History section placeholder */}
      <div className="mt-6 flex-1">
        <h3 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
          History
        </h3>
        <div className="space-y-1">
          {[1, 2, 3].map((i) => (
            <button
              key={i}
              className="w-full flex items-center gap-2 p-2 rounded text-sm text-muted-foreground hover:bg-background hover:text-foreground transition-colors"
            >
              <MessageSquare className="h-4 w-4" />
              <span className="truncate">Previous chat {i}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto pt-4 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          Press <kbd className="px-1 py-0.5 bg-muted rounded text-terminal-green">Ctrl+K</kbd> for shortcuts
        </p>
      </div>
    </motion.div>
  );
};
