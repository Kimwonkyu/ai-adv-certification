import { useState, useEffect } from 'react';
import { BookOpen, BarChart3, Settings, GraduationCap } from 'lucide-react';

interface HeaderProps {
    activeTab: 'dashboard' | 'study' | 'mistakes' | 'material' | 'hidden';
    setActiveTab: (tab: 'dashboard' | 'study' | 'mistakes' | 'material' | 'hidden') => void;
}

export function Header({ activeTab, setActiveTab }: HeaderProps) {
    const [clickCount, setClickCount] = useState(0);
    const [lastClickTime, setLastClickTime] = useState(0);

    const handleLogoClick = () => {
        const now = Date.now();

        // Reset if too slow (more than 1s between clicks)
        if (now - lastClickTime > 1000) {
            setClickCount(1);
        } else {
            const newCount = clickCount + 1;
            setClickCount(newCount);

            if (newCount >= 3) {
                // Short delay to show the final color before switching
                setTimeout(() => {
                    setActiveTab('hidden');
                    setClickCount(0);
                }, 300);
            }
        }
        setLastClickTime(now);
    };

    const getIconColor = () => {
        if (clickCount === 0) return 'bg-blue-600';
        if (clickCount === 1) return 'bg-emerald-500'; // First click: Green
        if (clickCount === 2) return 'bg-rose-600';    // Second click: Red
        return 'bg-blue-600'; // Reset/Action state
    };

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-xl border-b border-zinc-800">
            <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
                <div className="flex items-center gap-2 font-bold text-xl tracking-tight text-white select-none transition-transform">
                    <div
                        onClick={handleLogoClick}
                        className={`w-8 h-8 rounded-lg flex items-center justify-center cursor-pointer active:scale-90 transition-colors duration-200 ${getIconColor()}`}
                    >
                        <BookOpen className="w-5 h-5 text-white" />
                    </div>
                    <div className="hidden md:block">
                        AI <span className="text-zinc-500 font-medium">Certification</span>
                    </div>
                    <div className="md:hidden">
                        AI
                    </div>
                </div>

                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setActiveTab('dashboard')}
                        className={`flex items-center gap-2 px-3 md:px-4 py-2 rounded-full transition-all ${activeTab === 'dashboard' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                            }`}
                    >
                        <BarChart3 className="w-5 h-5 md:w-4 md:h-4" />
                        <span className="text-sm font-medium hidden md:block">대시보드</span>
                    </button>
                    <button
                        onClick={() => setActiveTab('study')}
                        className={`flex items-center gap-2 px-3 md:px-4 py-2 rounded-full transition-all ${activeTab === 'study' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                            }`}
                    >
                        <GraduationCap className="w-5 h-5 md:w-4 md:h-4" />
                        <span className="text-sm font-medium hidden md:block">학습하기</span>
                    </button>
                    <button
                        onClick={() => setActiveTab('material')}
                        className={`flex items-center gap-2 px-3 md:px-4 py-2 rounded-full transition-all ${activeTab === 'material' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                            }`}
                    >
                        <BookOpen className="w-5 h-5 md:w-4 md:h-4" />
                        <span className="text-sm font-medium hidden md:block">교재</span>
                    </button>
                </div>
            </div>
        </nav>
    );
}
