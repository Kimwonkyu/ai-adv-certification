'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import { Question, ConfidenceLevel } from '@/types';
import { Header } from '@/components/Header';
import { Dashboard } from '@/components/Dashboard';
import { StudyCard } from '@/components/StudyCard';
import { AuthGuard } from '@/components/AuthGuard';
import { Timer } from '@/components/Timer';
import { useProgress } from '@/hooks/useProgress';
import { cn } from '@/lib/utils';
import { Loader2, ArrowLeft, Trophy, Zap, Clock, TrendingUp } from 'lucide-react';
import { TextbookList, TEXTBOOKS } from '@/components/TextbookList';
import { TextbookViewer } from '@/components/TextbookViewer';
import { HiddenQuestionList } from '@/components/HiddenQuestionList';

const SESSION_KEY = 'vibe_active_session';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'study' | 'material' | 'hidden'>('dashboard');
  const [selectedTextbookId, setSelectedTextbookId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [allQuestions, setAllQuestions] = useState<Question[]>([]);
  const [studyPool, setStudyPool] = useState<Question[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [studyMode, setStudyMode] = useState<string>('');
  const [timerSeconds, setTimerSeconds] = useState(600);
  const [sessionStartTime, setSessionStartTime] = useState<number>(0);
  const [isResultView, setIsResultView] = useState(false);
  const [showSoftStop, setShowSoftStop] = useState(false);
  const [showModeAlert, setShowModeAlert] = useState(false);
  const [timeLogs, setTimeLogs] = useState<number[]>([]);
  const lastAtRef = useRef<number>(Date.now());

  const { progress, updateConfidence, resetProgress } = useProgress();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const response = await fetch('/questions.json');
      const data = await response.json();
      setAllQuestions(data);

      const savedSession = localStorage.getItem(SESSION_KEY);
      if (savedSession) {
        const session = JSON.parse(savedSession);
        setStudyPool(session.pool);
        setCurrentIdx(session.idx);
        setStudyMode(session.mode);
        setTimerSeconds(session.timer);
        setSessionStartTime(session.startTime || Date.now());
        setTimeLogs(session.logs || []);
      }
    } catch (error) {
      console.error('Failed to load questions:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveSession = (pool: Question[], idx: number, mode: string, timer: number, logs: number[]) => {
    localStorage.setItem(SESSION_KEY, JSON.stringify({
      pool,
      idx,
      mode,
      timer,
      startTime: sessionStartTime || Date.now(),
      logs
    }));
  };

  const clearSession = () => {
    localStorage.removeItem(SESSION_KEY);
    setStudyPool([]);
    setCurrentIdx(0);
    setStudyMode('');
    setTimeLogs([]);
    setIsResultView(false);
  };

  const chapters = useMemo(() => {
    const counts: Record<string, number> = {};
    const completedCounts: Record<string, number> = {};

    allQuestions.forEach(q => {
      counts[q.chapter_name] = (counts[q.chapter_name] || 0) + 1;
      if (progress.completed_ids.includes(q.id)) {
        completedCounts[q.chapter_name] = (completedCounts[q.chapter_name] || 0) + 1;
      }
    });

    return Object.entries(counts).map(([name, total]) => ({
      name,
      total,
      completed: completedCounts[name] || 0
    }));
  }, [allQuestions, progress.completed_ids]);

  const shuffleArray = <T,>(array: T[]): T[] => {
    const newArr = [...array];
    for (let i = newArr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [newArr[i], newArr[j]] = [newArr[j], newArr[i]];
    }
    return newArr;
  };

  const startChapterStudy = (selectedChapterNames: string[], shuffleOptions: boolean, count: number, type: 'all' | '객관식' | '코드 완성형' = 'all') => {
    let filtered = allQuestions.filter(q => selectedChapterNames.includes(q.chapter_name));

    if (type !== 'all') {
      filtered = filtered.filter(q => q.type === type);
    }

    filtered = shuffleArray(filtered).slice(0, count);

    if (shuffleOptions) {
      filtered = filtered.map(q => ({
        ...q,
        options: q.options ? shuffleArray(q.options) : q.options
      }));
    }

    const modeName = count === 999 ? "Pure Study" : "Intensive Study";
    setStudyPool(filtered);
    setStudyMode(`${modeName}: ${selectedChapterNames.join(', ')}`);
    setTimerSeconds(count === 999 ? 0 : 600);
    setSessionStartTime(Date.now());
    setTimeLogs([]);
    initStudy(filtered, `${modeName}: ${selectedChapterNames.join(', ')}`, count === 999 ? 0 : 600);
  };

  const startMockExam = (shuffleOptions: boolean) => {
    const mcqs = shuffleArray(allQuestions.filter(q => q.type === '객관식')).slice(0, 100);
    const subjective = shuffleArray(allQuestions.filter(q => q.type === '코드 완성형')).slice(0, 10);

    let pool = [...mcqs, ...subjective];
    pool = shuffleArray(pool);

    if (shuffleOptions) {
      pool = pool.map(q => ({
        ...q,
        options: q.options ? shuffleArray(q.options) : q.options
      }));
    }

    setStudyPool(pool);
    setStudyMode('AI ADV. 실전 모의고사 (100+10)');
    setTimerSeconds(6000);
    setSessionStartTime(Date.now());
    setTimeLogs([]);
    initStudy(pool, 'AI ADV. 실전 모의고사 (100+10)', 6000);
  };

  const startWeakness = (shuffleOptions: boolean) => {
    let filtered = allQuestions.filter(q => {
      const conf = progress.confidence_map[q.id];
      return conf === 'again' || conf === 'hard';
    });

    if (filtered.length === 0) {
      alert('현재 취약 문항이 없습니다! 모의고사 모드에 도전해보세요.');
      return;
    }

    if (shuffleOptions) {
      filtered = filtered.map(q => ({
        ...q,
        options: q.options ? shuffleArray(q.options) : q.options
      }));
    }

    const selectedPool = shuffleArray(filtered).slice(0, 10);
    setStudyPool(selectedPool);
    setStudyMode('Weakness Targeting');
    setTimerSeconds(600);
    setSessionStartTime(Date.now());
    setTimeLogs([]);
    initStudy(selectedPool, 'Weakness Targeting', 600);
  };

  const startReview = (shuffleOptions: boolean) => {
    let filtered = allQuestions.filter(q => {
      const conf = progress.confidence_map[q.id];
      return conf === 'good';
    });

    if (filtered.length === 0) {
      alert('현재 복습할 문항이 없습니다! 먼저 학습을 통해 실력을 다져보세요.');
      return;
    }

    if (shuffleOptions) {
      filtered = filtered.map(q => ({
        ...q,
        options: q.options ? shuffleArray(q.options) : q.options
      }));
    }

    const selectedPool = shuffleArray(filtered).slice(0, 10);
    setStudyPool(selectedPool);
    setStudyMode('Spaced Review');
    setTimerSeconds(600);
    setSessionStartTime(Date.now());
    setTimeLogs([]);
    initStudy(selectedPool, 'Spaced Review', 600);
  };

  const initStudy = (pool: Question[], mode: string, timer: number) => {
    setCurrentIdx(0);
    setIsResultView(false);
    setActiveTab('study');
    lastAtRef.current = Date.now();
    saveSession(pool, 0, mode, timer, []);
  };

  const handleResume = () => {
    setActiveTab('study');
    lastAtRef.current = Date.now();
  };

  const handleFeedback = (level: ConfidenceLevel) => {
    const currentQuestion = studyPool[currentIdx];
    updateConfidence(currentQuestion.id, level);

    const now = Date.now();
    const timeTaken = Math.round((now - lastAtRef.current) / 1000);
    const newLogs = [...timeLogs, timeTaken];
    setTimeLogs(newLogs);
    lastAtRef.current = now;

    if (currentIdx < studyPool.length - 1) {
      const nextIdx = currentIdx + 1;
      setCurrentIdx(nextIdx);
      saveSession(studyPool, nextIdx, studyMode, timerSeconds, newLogs);
    } else {
      setIsResultView(true);
      localStorage.removeItem(SESSION_KEY);
    }
  };

  const calculateAverageTime = () => {
    if (timeLogs.length === 0) return 0;
    const sum = timeLogs.reduce((a, b) => a + b, 0);
    return Math.round(sum / timeLogs.length);
  };

  const getBehavioralFeedback = (avg: number) => {
    if (avg < 20) return {
      title: "너무 서두르지 않았나요?",
      desc: "문제당 평균 20초 미만입니다. 정답을 맞히는 것보다 '왜' 정답인지 이해하는 깊은 처리(Deep Processing)가 중요합니다.",
      color: "text-rose-400"
    };
    if (avg < 50) return {
      title: "아주 적절한 깊이입니다!",
      desc: `평균 ${avg}초 소요. 핵심 개념을 인출하고 검토하기에 최적화된 속도로 학습하셨습니다.`,
      color: "text-emerald-400"
    };
    return {
      title: "정말 꼼꼼하게 학습하셨네요!",
      desc: "하나의 문항도 놓치지 않고 분석하셨습니다. 이 정독 습관이 실전에서 실수를 줄여줄 것입니다.",
      color: "text-blue-400"
    };
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0a0a0a]">
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
      </div>
    );
  }

  return (
    <AuthGuard>
      <main className="min-h-screen pt-24 pb-20 px-6 bg-[#0a0a0a] text-zinc-100">
        <Header activeTab={activeTab as any} setActiveTab={(tab) => {
          if (tab === 'dashboard') {
            setActiveTab('dashboard');
          } else if (tab === 'material') {
            setActiveTab('material');
            setSelectedTextbookId(null);
          } else if (tab === 'study') {
            if (studyPool.length > 0 && currentIdx < studyPool.length) {
              setActiveTab('study');
            } else {
              setShowModeAlert(true);
            }
          } else if (tab === 'hidden') {
            setActiveTab('hidden');
          }
        }} />

        <div className="max-w-5xl mx-auto pt-6">
          {activeTab === 'hidden' ? (
            <HiddenQuestionList
              questions={allQuestions}
              onBack={() => setActiveTab('dashboard')}
            />
          ) : activeTab === 'dashboard' ? (
            <Dashboard
              progress={progress}
              chapters={chapters}
              onStartStudy={startChapterStudy}
              onStartMockExam={startMockExam}
              onStartWeakness={startWeakness}
              onStartReview={startReview}
              onResetProgress={() => {
                resetProgress();
                clearSession();
              }}
              onResetSession={clearSession}
              hasActiveSession={studyPool.length > 0 && currentIdx < studyPool.length}
              onResumeStudy={handleResume}
            />
          ) : activeTab === 'material' ? (
            selectedTextbookId ? (
              <TextbookViewer
                chapterId={selectedTextbookId}
                title={TEXTBOOKS.find(b => b.id === selectedTextbookId)?.title || ''}
                onBack={() => setSelectedTextbookId(null)}
              />
            ) : (
              <TextbookList onSelect={setSelectedTextbookId} />
            )
          ) : isResultView ? (
            <div className="animate-in zoom-in-95 duration-700 max-w-2xl mx-auto py-10">
              <div className="glass-card p-10 border-emerald-500/20 bg-emerald-500/5 text-center">
                <div className="w-20 h-20 bg-emerald-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-emerald-500/30">
                  <Trophy className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-4xl font-black text-white uppercase italic tracking-tighter mb-2">Session Complete!</h2>
                <p className="text-zinc-400 font-bold tracking-widest uppercase mb-10">오늘의 학습이 뇌에 성공적으로 각인되었습니다.</p>

                <div className="grid grid-cols-2 gap-4 mb-10">
                  <div className="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <span className="block text-zinc-500 text-[10px] uppercase font-black tracking-widest mb-2">평균 소요 시간</span>
                    <div className="flex items-center justify-center gap-2">
                      <Clock className="w-5 h-5 text-blue-500" />
                      <span className="text-3xl font-black text-white">{calculateAverageTime()}s</span>
                    </div>
                  </div>
                  <div className="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <span className="block text-zinc-500 text-[10px] uppercase font-black tracking-widest mb-2">학습 효율도</span>
                    <div className="flex items-center justify-center gap-2">
                      <TrendingUp className="w-5 h-5 text-emerald-500" />
                      <span className="text-3xl font-black text-white">{Math.round((studyPool.length / 110) * 100)}%</span>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-500/5 border border-blue-500/20 p-6 rounded-2xl mb-10">
                  <p className="text-sm font-bold text-blue-400 mb-1">시험 통과 기준 가이드</p>
                  <p className="text-zinc-400 text-xs">본 모의고사는 총 110문항으로 구성되어 있습니다. <br />실제 시험 기준 80점 이상을 획득해야 합격입니다.</p>
                </div>

                <div className={cn("p-6 rounded-2xl border mb-10 text-left", getBehavioralFeedback(calculateAverageTime()).color.replace('text', 'bg').replace('400', '500/10') + " " + getBehavioralFeedback(calculateAverageTime()).color.replace('text', 'border').replace('400', '500/30'))}>
                  <div className="flex items-center gap-3 mb-2">
                    <Zap className={cn("w-5 h-5", getBehavioralFeedback(calculateAverageTime()).color)} />
                    <h4 className={cn("font-black text-lg", getBehavioralFeedback(calculateAverageTime()).color)}>
                      {getBehavioralFeedback(calculateAverageTime()).title}
                    </h4>
                  </div>
                  <p className="text-zinc-300 leading-relaxed text-sm">
                    {getBehavioralFeedback(calculateAverageTime()).desc}
                  </p>
                </div>

                <button
                  onClick={() => {
                    clearSession();
                    setActiveTab('dashboard');
                  }}
                  className="w-full py-4 bg-zinc-100 hover:bg-white text-black rounded-2xl font-black uppercase tracking-widest transition-all active:scale-95"
                >
                  대시보드로 돌아가기
                </button>
              </div>
            </div>
          ) : (
            <div className="animate-in fade-in duration-700 max-w-4xl mx-auto">
              <div className="mb-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => setActiveTab('dashboard')}
                    className="p-2 hover:bg-zinc-800 rounded-full transition-colors text-zinc-500 hover:text-white"
                  >
                    <ArrowLeft className="w-6 h-6" />
                  </button>
                  <div>
                    <h2 className="text-xl font-black uppercase italic tracking-tighter text-blue-500">{studyMode}</h2>
                    <p className="text-xs text-zinc-500 font-bold tracking-widest uppercase">
                      Question {currentIdx + 1} of {studyPool.length}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  {timerSeconds > 0 && (
                    <Timer key={studyMode + timerSeconds} initialSeconds={timerSeconds} onTimeUp={() => {
                      setShowSoftStop(true);
                    }} />
                  )}
                  <div className="w-32 h-2 bg-zinc-800 rounded-full overflow-hidden border border-white/5">
                    <div
                      className="h-full bg-emerald-500 transition-all duration-500"
                      style={{ width: `${((currentIdx + 1) / studyPool.length) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {showSoftStop && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[200] flex items-center justify-center p-6 animate-in fade-in duration-300">
                  <div className="glass-card max-w-md w-full p-8 border-rose-500/20 bg-zinc-900 shadow-2xl">
                    <div className="w-16 h-16 bg-rose-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                      <Clock className="w-8 h-8 text-rose-500" />
                    </div>
                    <h3 className="text-2xl font-black text-white text-center mb-2 uppercase italic">10분을 초과했습니다!</h3>
                    <p className="text-zinc-400 text-center text-sm leading-relaxed mb-8">
                      계획한 학습 시간이 모두 경과되었습니다.<br />남은 문제를 더 푸시겠습니까, 아니면 종료하시겠습니까?
                    </p>
                    <div className="grid grid-cols-2 gap-4">
                      <button
                        onClick={() => setShowSoftStop(false)}
                        className="py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl font-bold transition-all"
                      >
                        더 공부하기
                      </button>
                      <button
                        onClick={() => {
                          setShowSoftStop(false);
                          setIsResultView(true);
                          localStorage.removeItem(SESSION_KEY);
                        }}
                        className="py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-xl font-bold shadow-lg shadow-rose-500/20 transition-all"
                      >
                        여기서 종료
                      </button>
                    </div>
                  </div>
                </div>
              )}

              <StudyCard
                key={studyPool[currentIdx].id}
                question={studyPool[currentIdx]}
                onFeedback={handleFeedback}
              />
            </div>
          )}
        </div>

        {activeTab === 'dashboard' && progress.mastered_ids.length > 0 && (
          <div className="fixed bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-zinc-900/90 backdrop-blur-xl border border-emerald-500/20 px-6 py-3 rounded-2xl shadow-2xl z-[100]">
            <Trophy className="w-4 h-4 text-emerald-500" />
            <span className="text-sm font-bold text-zinc-300">
              총 <span className="text-emerald-400 font-black">{progress.mastered_ids.length}개의 문항</span>을 마스터했습니다.
            </span>
          </div>
        )}

        {showModeAlert && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[200] flex items-center justify-center p-6 animate-in fade-in duration-300">
            <div className="glass-card max-w-sm w-full p-8 border-amber-500/20 bg-zinc-900 shadow-2xl">
              <div className="w-16 h-16 bg-amber-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <Zap className="w-8 h-8 text-amber-500" />
              </div>
              <h3 className="text-xl font-black text-white text-center mb-2 uppercase italic">학습 모드 선택 필요</h3>
              <p className="text-zinc-400 text-center text-sm leading-relaxed mb-8">
                먼저 대시보드에서 원하는 학습 모드를<br />선택해주세요!
              </p>
              <button
                onClick={() => setShowModeAlert(false)}
                className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl font-bold transition-all"
              >
                확인
              </button>
            </div>
          </div>
        )}
      </main>
    </AuthGuard>
  );
}
