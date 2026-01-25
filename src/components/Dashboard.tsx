import { useState } from 'react';
import { Shuffle, Target, Layers, PlayCircle, CheckCircle2 } from 'lucide-react';
import { UserProgress } from '@/types';
import { cn } from '@/lib/utils';

interface ChapterProgress {
    name: string;
    total: number;
    completed: number;
}

interface DashboardProps {
    progress: UserProgress;
    chapters: ChapterProgress[];
    onStartStudy: (selectedChapters: string[], shuffleOptions: boolean, count: number, type?: 'all' | '객관식' | '주관식') => void;
    onStartMockExam: (shuffleOptions: boolean) => void;
    onStartWeakness: (shuffleOptions: boolean) => void;
    onResetProgress: () => void;
    onResetSession: () => void;
    hasActiveSession: boolean;
    onResumeStudy: () => void;
}

export function Dashboard({
    progress,
    chapters,
    onStartStudy,
    onStartMockExam,
    onStartWeakness,
    onResetProgress,
    onResetSession,
    hasActiveSession,
    onResumeStudy
}: DashboardProps) {
    const [selectedChapters, setSelectedChapters] = useState<string[]>([]);
    const [shuffleEnabled, setShuffleEnabled] = useState(true);
    const [studyMode, setStudyMode] = useState<'deep' | 'speed' | 'custom' | 'pure'>('deep');
    const [customCount, setCustomCount] = useState(5);
    const [pureTarget, setPureTarget] = useState<'all' | '객관식' | '주관식'>('all');
    const [showResetConfirm, setShowResetConfirm] = useState(false);
    const [showProgressResetConfirm, setShowProgressResetConfirm] = useState(false);

    const totalQuestions = chapters.reduce((acc, c) => acc + c.total, 0);
    const totalCompleted = progress.completed_ids.length;
    const masteredCount = progress.mastered_ids.length;
    const progressPercent = Math.round((totalCompleted / totalQuestions) * 100);

    const toggleChapter = (name: string) => {
        setSelectedChapters(prev =>
            prev.includes(name) ? prev.filter(c => c !== name) : [...prev, name]
        );
    };

    return (
        <div className="space-y-10 animate-in fade-in slide-in-from-bottom-6 duration-1000">
            {/* 1. Statistics Overview */}
            <div className="glass-card bg-gradient-to-br from-blue-600/20 to-zinc-900/50 border-blue-500/20">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
                    <div>
                        <h2 className="text-3xl font-black text-white mb-2 tracking-tighter uppercase italic">AI ADV. 모의고사 <span className="text-blue-500 not-italic ml-2">Dashboard</span></h2>
                        <p className="text-zinc-400 font-medium">실전을 위한 모의고사 훈련을 진행하세요.</p>
                    </div>
                    <div className="flex items-center gap-4 bg-black/40 p-4 rounded-2xl border border-white/5">
                        <div className="text-center px-4">
                            <span className="block text-2xl font-black text-blue-500">{progressPercent}%</span>
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest px-1">Progress</span>
                        </div>
                        <div className="w-px h-10 bg-zinc-800" />
                        <div className="text-center px-4">
                            <span className="block text-2xl font-black text-emerald-500">{masteredCount}</span>
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest px-1">Mastered</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-between mb-4">
                    <div className="flex-grow mr-4">
                        <div className="w-full h-4 bg-zinc-800/50 rounded-full overflow-hidden mb-2 p-1">
                            <div
                                className="h-full bg-gradient-to-r from-blue-600 to-emerald-400 rounded-full transition-all duration-[2000ms] shadow-[0_0_15px_rgba(59,130,246,0.3)]"
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                    </div>
                    <button
                        onClick={() => setShowProgressResetConfirm(true)}
                        className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-rose-400 border border-zinc-700 rounded-lg text-xs font-bold transition-all flex items-center gap-2"
                    >
                        <Shuffle className="w-3 h-3 rotate-45" /> 초기화
                    </button>
                </div>
            </div>

            {/* 2. Core Study Modes */}
            <div className="flex flex-col gap-4 mb-6">
                <div className="flex items-center justify-between p-4 glass-card border-zinc-800 bg-zinc-900/30">
                    <div className="flex items-center gap-3">
                        <Shuffle className="w-5 h-5 text-zinc-500" />
                        <div>
                            <p className="text-sm font-bold text-white uppercase italic">보기 순서 무작위 섞기</p>
                            <p className="text-[10px] text-zinc-500 font-medium">객관식 문항의 보기 위치를 섞어 단순 암기를 방지합니다.</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setShuffleEnabled(!shuffleEnabled)}
                        className={cn(
                            "w-12 h-6 rounded-full p-1 transition-all duration-300",
                            shuffleEnabled ? "bg-blue-600" : "bg-zinc-700"
                        )}
                    >
                        <div className={cn(
                            "w-4 h-4 bg-white rounded-full transition-transform duration-300",
                            shuffleEnabled ? "translate-x-6" : "translate-x-0"
                        )} />
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <button
                        onClick={() => onStartMockExam(shuffleEnabled)}
                        className="group relative overflow-hidden glass-card h-52 border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10 hover:border-emerald-500/60 transition-all text-left"
                    >
                        <div className="absolute -right-6 -bottom-6 opacity-10 group-hover:opacity-20 transition-opacity">
                            <Shuffle className="w-32 h-32 text-emerald-500" />
                        </div>
                        <div className="relative z-10 flex flex-col h-full justify-between">
                            <div>
                                <div className="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-emerald-500/20">
                                    <Shuffle className="w-5 h-5 text-white" />
                                </div>
                                <h3 className="text-xl font-black text-white mb-2 uppercase italic">전체 실전 모의고사</h3>
                                <p className="text-sm text-zinc-400 leading-relaxed max-w-[280px]">
                                    60분 타이머 적용. 전 단원 문항을 무작위로 섞어서 실전과 동일한 환경으로 테스트합니다.
                                </p>
                            </div>
                            <div className="text-xs font-bold text-emerald-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                실전 모드 시작 (1시간) →
                            </div>
                        </div>
                    </button>

                    <button
                        onClick={() => onStartWeakness(shuffleEnabled)}
                        className="group relative overflow-hidden glass-card h-52 border-rose-500/30 bg-rose-500/5 hover:bg-rose-500/10 hover:border-rose-500/60 transition-all text-left"
                    >
                        <div className="absolute -right-6 -bottom-6 opacity-10 group-hover:opacity-20 transition-opacity">
                            <Target className="w-32 h-32 text-rose-500" />
                        </div>
                        <div className="relative z-10 flex flex-col h-full justify-between">
                            <div>
                                <div className="w-10 h-10 bg-rose-500 rounded-xl flex items-center justify-center mb-4 shadow-lg shadow-rose-500/20">
                                    <Target className="w-5 h-5 text-white" />
                                </div>
                                <h3 className="text-xl font-black text-white mb-2 uppercase italic">오답/약점 격파</h3>
                                <p className="text-sm text-zinc-400 leading-relaxed max-w-[280px]">
                                    &apos;몰랐음&apos;이나 &apos;어려움&apos;으로 평가한 문항만 모아서 반복 학습합니다. (10분 집중)
                                </p>
                            </div>
                            <div className="text-xs font-bold text-rose-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                취약점 보완하기 (10분) →
                            </div>
                        </div>
                    </button>
                </div>
            </div>

            {/* 3. Personalized Study Configuration */}
            <div className="glass-card p-8 border-zinc-800 bg-zinc-900/40 space-y-8">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <PlayCircle className="w-6 h-6 text-blue-500" />
                        <div>
                            <h3 className="text-xl font-black text-white uppercase italic">학습 목표 설정</h3>
                            <p className="text-xs text-zinc-500 font-bold uppercase tracking-widest">목표 문항 수와 학습 깊이를 선택하세요.</p>
                        </div>
                    </div>
                    {hasActiveSession && (
                        <div className="flex items-center gap-3">
                            <button
                                onClick={onResumeStudy}
                                className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-black text-sm shadow-lg shadow-emerald-500/20 transition-all active:scale-95 flex items-center gap-2"
                            >
                                <PlayCircle className="w-4 h-4" /> 이어 학습하기
                            </button>
                            <button
                                onClick={() => setShowResetConfirm(true)}
                                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 rounded-xl font-bold text-xs transition-all flex items-center gap-2"
                            >
                                <Shuffle className="w-3 h-3" /> 세션 초기화
                            </button>
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <button
                        onClick={() => setStudyMode('deep')}
                        className={cn(
                            "p-6 rounded-2xl border transition-all text-left group",
                            studyMode === 'deep' ? "border-blue-500 bg-blue-500/10 shadow-[0_0_20px_rgba(59,130,246,0.1)]" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                        )}
                    >
                        <div className="flex items-center gap-3 mb-3">
                            <div className={cn("w-2 h-2 rounded-full", studyMode === 'deep' ? "bg-blue-500 animate-pulse" : "bg-zinc-700")} />
                            <h4 className="font-black text-white uppercase italic">정독 모드</h4>
                        </div>
                        <p className="text-[10px] text-zinc-400 font-bold mb-4 uppercase tracking-tighter leading-relaxed">Deep Dive (3~5문항 / 10분)</p>
                        <p className="text-xs text-zinc-500 leading-relaxed font-medium">코드와 해설을 꼼꼼히 분석합니다. 새로운 개념 학습에 추천합니다.</p>
                    </button>

                    <button
                        onClick={() => setStudyMode('speed')}
                        className={cn(
                            "p-6 rounded-2xl border transition-all text-left group",
                            studyMode === 'speed' ? "border-emerald-500 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.1)]" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                        )}
                    >
                        <div className="flex items-center gap-3 mb-3">
                            <div className={cn("w-2 h-2 rounded-full", studyMode === 'speed' ? "bg-emerald-500 animate-pulse" : "bg-zinc-700")} />
                            <h4 className="font-black text-white uppercase italic">스피드 모드</h4>
                        </div>
                        <p className="text-[10px] text-zinc-400 font-bold mb-4 uppercase tracking-tighter leading-relaxed">Speed Run (10~12문항 / 10분)</p>
                        <p className="text-xs text-zinc-500 leading-relaxed font-medium">이미 아는 내용을 빠르게 복습합니다. 인출 훈련에 추천합니다.</p>
                    </button>

                    <button
                        onClick={() => setStudyMode('custom')}
                        className={cn(
                            "p-6 rounded-2xl border transition-all text-left group",
                            studyMode === 'custom' ? "border-rose-500 bg-rose-500/10 shadow-[0_0_20px_rgba(244,63,94,0.1)]" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                        )}
                    >
                        <div className="flex items-center gap-3 mb-3">
                            <div className={cn("w-2 h-2 rounded-full", studyMode === 'custom' ? "bg-rose-500 animate-pulse" : "bg-zinc-700")} />
                            <h4 className="font-black text-white uppercase italic">사용자 지정</h4>
                        </div>
                        <p className="text-[10px] text-zinc-400 font-bold mb-4 uppercase tracking-tighter leading-relaxed">Custom Count ({customCount}문항 / 10분)</p>
                        <p className="text-xs text-zinc-500 leading-relaxed font-medium">원하는 문제 수를 직접 설정하여 학습량을 조절합니다.</p>
                    </button>

                    <button
                        onClick={() => setStudyMode('pure')}
                        className={cn(
                            "p-6 rounded-2xl border transition-all text-left group",
                            studyMode === 'pure' ? "border-amber-500 bg-amber-500/10 shadow-[0_0_20px_rgba(245,158,11,0.1)]" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                        )}
                    >
                        <div className="flex items-center gap-3 mb-3">
                            <div className={cn("w-2 h-2 rounded-full", studyMode === 'pure' ? "bg-amber-500 animate-pulse" : "bg-zinc-700")} />
                            <h4 className="font-black text-white uppercase italic">순수 학습 모드</h4>
                        </div>
                        <p className="text-[10px] text-zinc-400 font-bold mb-4 uppercase tracking-tighter leading-relaxed">Unlimited (시간 제약 없음)</p>
                        <p className="text-xs text-zinc-500 leading-relaxed font-medium">시간 압박 없이 유형별 문제를 집중적으로 학습합니다.</p>
                    </button>
                </div>

                {studyMode === 'custom' && (
                    <div className="p-6 bg-black/40 rounded-2xl border border-rose-500/20 animate-in slide-in-from-top-4 duration-500">
                        <div className="flex items-center justify-between mb-6">
                            <span className="text-sm font-black text-zinc-400 uppercase tracking-widest">문항 수: <span className="text-rose-400">{customCount}</span></span>
                            {customCount >= 15 && (
                                <div className="flex items-center gap-2 text-rose-400 animate-pulse">
                                    <Target className="w-4 h-4" />
                                    <span className="text-[10px] font-black uppercase tracking-tighter">문제당 40초 미만입니다. 충분히 깊은 학습이 가능할까요?</span>
                                </div>
                            )}
                        </div>
                        <input
                            type="range"
                            min="1"
                            max="20"
                            value={customCount}
                            onChange={(e) => setCustomCount(parseInt(e.target.value))}
                            className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                        />
                    </div>
                )}

                {studyMode === 'pure' && (
                    <div className="p-6 bg-black/40 rounded-2xl border border-amber-500/20 animate-in slide-in-from-top-4 duration-500">
                        <p className="text-sm font-black text-zinc-400 uppercase tracking-widest mb-4">학습 문항 유형 선택</p>
                        <div className="flex gap-4">
                            {(['all', '객관식', '주관식'] as const).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => setPureTarget(type)}
                                    className={cn(
                                        "px-6 py-2.5 rounded-xl text-sm font-bold transition-all",
                                        pureTarget === type ? "bg-amber-600 text-white" : "bg-zinc-800 text-zinc-500 hover:bg-zinc-700"
                                    )}
                                >
                                    {type === 'all' ? '전체' : type}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* 4. Multi-Chapter Selection */}
            <div className="space-y-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
                    <div className="flex items-center gap-2">
                        <Layers className="w-5 h-5 text-zinc-500" />
                        <h3 className="text-xs font-black text-zinc-500 uppercase tracking-[0.3em]">단원별 맞춤 학습 (중복 선택 가능)</h3>
                    </div>
                    <button
                        disabled={selectedChapters.length === 0}
                        onClick={() => {
                            let count = studyMode === 'deep' ? 5 : studyMode === 'speed' ? 12 : customCount;
                            if (studyMode === 'pure') count = 999; // Signal for all available
                            onStartStudy(selectedChapters, shuffleEnabled, count, studyMode === 'pure' ? pureTarget : 'all');
                        }}
                        className={cn(
                            "flex items-center gap-2 px-6 py-2 rounded-xl font-bold text-sm transition-all active:scale-95",
                            selectedChapters.length > 0 ? "bg-blue-600 text-white shadow-lg shadow-blue-500/30" : "bg-zinc-800 text-zinc-600 cursor-not-allowed"
                        )}
                    >
                        <PlayCircle className="w-4 h-4" />
                        선택한 {selectedChapters.length}개 단원 학습 시작
                    </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {chapters.map((chapter) => {
                        const isSelected = selectedChapters.includes(chapter.name);
                        const percent = Math.round((chapter.completed / chapter.total) * 100);

                        return (
                            <div
                                key={chapter.name}
                                onClick={() => toggleChapter(chapter.name)}
                                className={cn(
                                    "glass-card p-5 group transition-all cursor-pointer relative overflow-hidden",
                                    isSelected ? "border-blue-500/60 bg-blue-500/5" : "hover:border-zinc-700"
                                )}
                            >
                                {isSelected && (
                                    <div className="absolute top-2 right-2 text-blue-500 animate-in zoom-in duration-300">
                                        <CheckCircle2 className="w-5 h-5 fill-blue-500 text-black" />
                                    </div>
                                )}
                                <div className="flex justify-between items-start mb-3">
                                    <h4 className={cn(
                                        "font-bold text-sm transition-colors",
                                        isSelected ? "text-blue-400" : "text-zinc-200 group-hover:text-white"
                                    )}>
                                        {chapter.name}
                                    </h4>
                                </div>
                                <div className="w-full h-1 bg-zinc-800 rounded-full mb-4">
                                    <div
                                        className={cn(
                                            "h-full transition-all duration-1000",
                                            isSelected ? "bg-blue-500" : "bg-zinc-600 group-hover:bg-zinc-400"
                                        )}
                                        style={{ width: `${percent}%` }}
                                    />
                                </div>
                                <div className="flex justify-between items-center text-[10px] font-black uppercase">
                                    <span className="text-zinc-600 tracking-widest">{chapter.total}문항</span>
                                    <span className={isSelected ? "text-blue-500" : "text-zinc-500"}>{percent}% Complete</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Confirmation Modals */}
            {showResetConfirm && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6 animate-in fade-in duration-200">
                    <div className="glass-card max-w-sm w-full p-6 border-zinc-700 bg-zinc-900 shadow-2xl">
                        <h3 className="text-lg font-black text-white mb-2">세션 초기화</h3>
                        <p className="text-zinc-400 text-sm mb-6 leading-relaxed">
                            현재 진행 중인 학습 데이터가 모두 사라집니다.<br />초기화 하시겠습니까?
                        </p>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowResetConfirm(false)}
                                className="flex-1 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg font-bold text-sm transition-all"
                            >
                                취소
                            </button>
                            <button
                                onClick={() => {
                                    onResetSession();
                                    setShowResetConfirm(false);
                                }}
                                className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-bold text-sm shadow-lg shadow-rose-500/20 transition-all"
                            >
                                초기화
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {showProgressResetConfirm && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6 animate-in fade-in duration-200">
                    <div className="glass-card max-w-sm w-full p-6 border-zinc-700 bg-zinc-900 shadow-2xl">
                        <h3 className="text-lg font-black text-white mb-2">전체 기록 초기화</h3>
                        <p className="text-zinc-400 text-sm mb-6 leading-relaxed">
                            모든 학습 이력과 마스터 기록이 영구적으로 삭제됩니다.<br />정말 초기화 하시겠습니까?
                        </p>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowProgressResetConfirm(false)}
                                className="flex-1 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg font-bold text-sm transition-all"
                            >
                                취소
                            </button>
                            <button
                                onClick={() => {
                                    onResetProgress();
                                    setShowProgressResetConfirm(false);
                                }}
                                className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg font-bold text-sm shadow-lg shadow-rose-500/20 transition-all"
                            >
                                전체 초기화
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
