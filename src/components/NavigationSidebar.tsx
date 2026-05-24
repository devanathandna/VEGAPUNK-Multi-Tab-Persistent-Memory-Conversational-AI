import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Home, LayoutDashboard, Brain, Settings, ChevronLeft, ChevronRight, History, Plus, Trash2, Edit2, Check, X } from "lucide-react";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Input } from "./ui/input";

interface Group {
  group_key: string;
  title: string;
  message_count: number;
  sessions: Array<{session_id: string, updated_at: string}>;
  updated_at: string;
}

interface NavigationSidebarProps {
  onNewSubChat: () => void;
  onNewChat: () => void;
  activeSubChats: number;
  maxSubChats: number;
  onGroupSelect: (groupKey: string) => void;
  refreshKey?: number;
}

export const NavigationSidebar = ({ onNewSubChat, onNewChat, activeSubChats, maxSubChats, onGroupSelect, refreshKey = 0 }: NavigationSidebarProps) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [groups, setGroups] = useState<Group[]>([]);
  const [editingGroup, setEditingGroup] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { icon: Home, label: "Home", path: "/" },
    { icon: LayoutDashboard, label: "VEGACHAT", path: "/vegachat" },
    { icon: Brain, label: "Seraphims", path: "/seraphims" },
    { icon: Settings, label: "Settings", path: "/settings" },
  ];

  useEffect(() => {
    // Fetch on mount AND whenever refreshKey changes (i.e. after any chat response)
    fetchGroups();
  }, [refreshKey]);  // eslint-disable-line react-hooks/exhaustive-deps

  const fetchGroups = async () => {
    try {
      // First check if backend is ready
      const healthResponse = await fetch('http://localhost:5000/api/health');
      const healthData = await healthResponse.json();
      
      if (!healthData.ready) {
        console.log('⏳ Backend is initializing...');
        return;
      }
      
      const response = await fetch('http://localhost:5000/api/sessions/grouped');
      if (response.ok) {
        const data = await response.json();
        setGroups(data.groups || []);
        console.log('✅ Fetched groups:', data.groups?.length || 0);
      } else {
        console.warn('⚠️ Failed to fetch groups, status:', response.status);
      }
    } catch (error) {
      console.error('❌ Failed to fetch groups:', error);
      // Silently fail - don't show error to user
    }
  };

  const handleDeleteGroup = async (groupKey: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this chat history?')) {
      try {
        const response = await fetch(`http://localhost:5000/api/sessions/group/${encodeURIComponent(groupKey)}`, {
          method: 'DELETE',
        });
        if (response.ok) {
          fetchGroups();
        }
      } catch (error) {
        console.error('Failed to delete group:', error);
      }
    }
  };

  const handleEditGroup = (groupKey: string, currentTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingGroup(groupKey);
    setEditTitle(currentTitle);
  };

  const handleSaveEdit = async (groupKey: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await fetch(`http://localhost:5000/api/sessions/group/${encodeURIComponent(groupKey)}/title`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_title: editTitle }),
      });
      if (response.ok) {
        setEditingGroup(null);
        fetchGroups();
      }
    } catch (error) {
      console.error('Failed to update title:', error);
    }
  };

  const handleCancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingGroup(null);
    setEditTitle("");
  };

  return (
    <div className={`h-screen bg-card border-r border-terminal-green/20 flex flex-col transition-all duration-300 ${isCollapsed ? "w-16" : "w-64"}`}>
      {/* Header */}
      <div className="p-4 border-b border-terminal-green/20 flex items-center justify-between">
        {!isCollapsed && (
          <div>
            <h2 className="text-lg font-bold text-terminal-green">VEGAPUNK</h2>
            <p className="text-xs text-muted-foreground">AI Chat System</p>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="hover:bg-terminal-green/10 hover:text-terminal-green"
        >
          {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="p-2 border-b border-terminal-green/20">
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Button
              key={item.path}
              variant="ghost"
              onClick={() => navigate(item.path)}
              className={`w-full justify-start mb-1 ${isActive ? "bg-terminal-green/20 text-terminal-green" : "hover:bg-terminal-green/10 hover:text-terminal-green"} ${isCollapsed ? "px-2" : ""}`}
            >
              <item.icon className={`h-4 w-4 ${isCollapsed ? "" : "mr-2"}`} />
              {!isCollapsed && <span>{item.label}</span>}
            </Button>
          );
        })}
      </nav>

      {/* New Satellite Button */}
      <div className="p-2 border-b border-terminal-green/20 space-y-2">
        <Button
          onClick={onNewSubChat}
          disabled={activeSubChats >= maxSubChats}
          className={`w-full bg-terminal-green/20 hover:bg-terminal-green/30 text-terminal-green border border-terminal-green ${isCollapsed ? "px-2" : ""}`}
        >
          <Plus className={`h-4 w-4 ${isCollapsed ? "" : "mr-2"}`} />
          {!isCollapsed && <span>New Satellite ({activeSubChats}/{maxSubChats})</span>}
        </Button>
        {!isCollapsed && (
          <Button
          onClick={onNewChat}
          className="w-full bg-terminal-green/10 hover:bg-terminal-green/20 text-terminal-green border border-terminal-green/50"
        >
          <Plus className="h-4 w-4 mr-2" />
          <span>New Chat</span>
        </Button>
        )}
      </div>

      {/* Chat History */}
      {!isCollapsed && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-2 flex items-center justify-between border-b border-terminal-green/20">
            <div className="flex items-center gap-2 text-sm font-semibold text-terminal-green">
              <History className="h-4 w-4" />
              <span>History</span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowHistory(!showHistory)}
              className="h-6 w-6 hover:bg-terminal-green/10 hover:text-terminal-green"
            >
              {showHistory ? <ChevronLeft className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            </Button>
          </div>

          {showHistory && (
            <ScrollArea className="flex-1 p-2">
              {groups.length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-4">No chat history yet</p>
              ) : (
                groups.map((group) => (
                  <div
                    key={group.group_key}
                    onClick={() => onGroupSelect(group.group_key)}
                    className="w-full text-left p-2 mb-1 rounded hover:bg-terminal-green/10 transition-colors group cursor-pointer relative"
                  >
                    {editingGroup === group.group_key ? (
                      <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                        <Input
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          className="h-7 text-sm bg-background/50 border-terminal-green/30 focus:border-terminal-green"
                          autoFocus
                        />
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={(e) => handleSaveEdit(group.group_key, e)}
                          className="h-7 w-7 hover:bg-terminal-green/20 hover:text-terminal-green"
                        >
                          <Check className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={handleCancelEdit}
                          className="h-7 w-7 hover:bg-red-500/20 hover:text-red-500"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ) : (
                      <>
                        <div className="mb-1">
                          <p className="text-sm text-foreground group-hover:text-terminal-green">
                            {group.title}
                          </p>
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{group.message_count} msgs</span>
                            <span>•</span>
                            <span>{new Date(group.updated_at).toLocaleDateString()}</span>
                          </div>
                          <div className="flex gap-1">
                            <button
                              onClick={(e) => handleEditGroup(group.group_key, group.title, e)}
                              className="px-1.5 py-1 text-sm rounded bg-terminal-green/30 text-terminal-green border border-terminal-green/50"
                            >
                              <Edit2 className="h-3 w-3" />
                            </button>
                            <button
                              onClick={(e) => handleDeleteGroup(group.group_key, e)}
                              className="px-1.5 py-1 text-sm rounded bg-red-500/30 text-red-500 border border-red-500/50"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                ))
              )}
            </ScrollArea>
          )}
        </div>
      )}
    </div>
  );
};
