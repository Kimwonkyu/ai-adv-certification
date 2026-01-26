import { useState, useEffect } from 'react';
import { Lock, ArrowRight, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

export function AuthGuard({ children }: { children: React.ReactNode }) {
    const [password, setPassword] = useState('');
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [error, setError] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const auth = localStorage.getItem('vibe_auth');
        if (auth === 'true') {
            setIsAuthenticated(true);
        }
        setIsLoading(false);
    }, []);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (password === 'aifighting') {
            localStorage.setItem('vibe_auth', 'true');
            setIsAuthenticated(true);
            setError(false);
        } else {
            setError(true);
            setPassword('');
        }
    };

    if (isLoading) return null;

    if (isAuthenticated) {
        return <>{children}</>;
    }

    return (
        <div className="min-h-screen bg-black flex items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-black to-black">
            <div className="w-full max-w-md animate-in fade-in zoom-in duration-500">
                <div className="glass-card p-10 border-zinc-800 bg-zinc-900/40 relative overflow-hidden">
                    {/* Decorative background blur */}
                    <div className="absolute -top-24 -left-24 w-48 h-48 bg-blue-600/10 rounded-full blur-[80px]" />
                    <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-emerald-600/10 rounded-full blur-[80px]" />

                    <div className="relative z-10 flex flex-col items-center text-center">
                        <div className="w-20 h-20 bg-blue-600/10 rounded-2xl flex items-center justify-center mb-8 border border-blue-500/20 shadow-lg shadow-blue-500/10">
                            <Lock className="w-10 h-10 text-blue-500" />
                        </div>

                        <h1 className="text-3xl font-black text-white mb-3 tracking-tighter uppercase italic">
                            ACCESS <span className="text-blue-500 not-italic">LOCKED</span>
                        </h1>
                        <p className="text-zinc-500 text-sm font-bold uppercase tracking-widest mb-10">
                            부여받은 비밀번호를 입력하세요
                        </p>

                        <form onSubmit={handleSubmit} className="w-full space-y-4">
                            <div className="relative group">
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Enter Password"
                                    className={cn(
                                        "w-full bg-black/40 border-2 rounded-2xl py-4 px-6 text-white placeholder:text-zinc-700 outline-none transition-all font-mono tracking-[0.2em]",
                                        error
                                            ? "border-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.1)]"
                                            : "border-zinc-800 focus:border-blue-500/50 group-hover:border-zinc-700"
                                    )}
                                    autoFocus
                                />
                                {error && (
                                    <p className="text-rose-500 text-[10px] font-black uppercase mt-2 text-left ml-2 tracking-widest animate-in slide-in-from-top-1">
                                        비밀번호가 일치하지 않습니다
                                    </p>
                                )}
                            </div>

                            <button
                                type="submit"
                                className="w-full h-14 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl font-black text-sm uppercase tracking-widest shadow-lg shadow-blue-600/20 transition-all active:scale-95 flex items-center justify-center gap-2 group"
                            >
                                Unlock <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                            </button>
                        </form>

                        <div className="mt-12 flex items-center gap-2 text-zinc-600">
                            <ShieldCheck className="w-4 h-4" />
                            <span className="text-[10px] font-black uppercase tracking-widest">Secure Access Protocol</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
