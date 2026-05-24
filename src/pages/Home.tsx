import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";

const Home = () => {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create neural network effect
    const container = containerRef.current;
    if (!container) return;

    // Create canvas for neural network
    const canvas = document.createElement('canvas');
    canvas.className = 'absolute inset-0 w-full h-full';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d')!;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Create nodes for the network
    const nodes = Array.from({ length: 50 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5
    }));

    // Animation loop
    const animateCanvas = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.1)';
      ctx.fillStyle = 'rgba(74, 222, 128, 0.1)';

      // Update and draw nodes
      nodes.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off walls
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Draw connections
        nodes.forEach(otherNode => {
          const dx = otherNode.x - node.x;
          const dy = otherNode.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(otherNode.x, otherNode.y);
            ctx.stroke();
          }
        });

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, 2, 0, Math.PI * 2);
        ctx.fill();
      });

      requestAnimationFrame(animateCanvas);
    };
    animateCanvas();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (container.contains(canvas)) {
        container.removeChild(canvas);
      }
    };
  }, []);

  return (
    <div ref={containerRef} className="relative flex h-screen w-full bg-background items-center justify-center p-12 overflow-hidden">
      <div className="relative z-10 text-center space-y-8 max-w-2xl">
        <div className="logo-container space-y-4">
          <div className="relative mx-auto w-32 h-32 mb-8">
            <img
              src="/vegapunk_logo.png"
              alt="VEGAPUNK Logo"
              className="w-full h-full object-contain drop-shadow-lg"
            />
          </div>
          <h2 className="text-5xl font-bold text-terminal-green">
            VEGAPUNK
          </h2>
          <p className="text-xl text-muted-foreground">
            The genius who split his mind into satellites to think in parallel.
          </p>
        </div>
        <div className="space-y-4 pt-8">
          <button
            onClick={() => navigate("/vegachat")}
            className="w-full max-w-md mx-auto px-8 py-4 bg-terminal-green/10 border-2 border-terminal-green text-terminal-green rounded-lg font-semibold hover:bg-terminal-green hover:text-black transition-all duration-300"
          >
            Enter the Chat Interface
          </button>
          <p className="text-sm text-muted-foreground">
            Click above to enter the parallel mind system
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;