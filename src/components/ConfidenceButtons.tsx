import { ConfidenceLevel } from '@/types';
import { RefreshCw, BrainCircuit, CheckCircle2, Star } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ConfidenceButtonsProps {
    onSelect: (level: ConfidenceLevel) => void;
    disabled?: boolean;
}

export function ConfidenceButtons({ onSelect, disabled }: ConfidenceButtonsProps) {
    const buttons: { id: ConfidenceLevel; label: string; color: string; icon: any; description: string }[] = [
        { id: 'again', label: '몰랐음', color: 'bg-red-500 hover:bg-red-400', icon: RefreshCw, description: '다시 공부' },
        { id: 'hard', label: '어려움', color: 'bg-amber-600 hover:bg-amber-500', icon: BrainCircuit, description: '곧 복습' },
        { id: 'good', label: '알맞음', color: 'bg-emerald-600 hover:bg-emerald-500', icon: CheckCircle2, description: '내일 복습' },
        { id: 'easy', label: '쉬움', color: 'bg-blue-600 hover:bg-blue-500', icon: Star, description: '마스터' },
    ];

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 w-full">
            {buttons.map((btn) => (
                <button
                    key={btn.id}
                    disabled={disabled}
                    onClick={() => onSelect(btn.id)}
                    className={cn(
                        "flex flex-col items-center justify-center p-4 rounded-2xl transition-all active:scale-95 border border-white/5",
                        btn.color,
                        disabled ? "opacity-50 grayscale" : "shadow-lg shadow-black/20"
                    )}
                >
                    <btn.icon className="w-6 h-6 mb-2 text-white" />
                    <span className="text-sm font-bold text-white">{btn.label}</span>
                    <span className="text-[10px] text-white/70 uppercase tracking-tighter">{btn.description}</span>
                </button>
            ))}
        </div>
    );
}
