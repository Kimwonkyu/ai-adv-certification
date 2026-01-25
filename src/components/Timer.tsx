import { useState, useEffect } from 'react';
import { Timer as TimerIcon } from 'lucide-react';

interface TimerProps {
    initialSeconds: number;
    onTimeUp?: () => void;
}

export function Timer({ initialSeconds, onTimeUp }: TimerProps) {
    const [timeLeft, setTimeLeft] = useState(initialSeconds);

    useEffect(() => {
        if (timeLeft <= 0) {
            onTimeUp?.();
            return;
        }

        const timer = setInterval(() => {
            setTimeLeft(prev => prev - 1);
        }, 1000);

        return () => clearInterval(timer);
    }, [timeLeft, onTimeUp]);

    const minutes = Math.floor(timeLeft / 60);
    const seconds = timeLeft % 60;

    return (
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all ${timeLeft < 300 ? 'bg-red-500/10 border-red-500/30 text-red-500' : 'bg-zinc-800/50 border-zinc-700 text-zinc-400'
            }`}>
            <TimerIcon className={`w-4 h-4 ${timeLeft < 300 ? 'animate-pulse' : ''}`} />
            <span className="text-sm font-mono font-bold">
                {String(minutes).padStart(2, '0')}:{String(seconds).padStart(2, '0')}
            </span>
        </div>
    );
}
