import { useEffect, useState, useRef } from 'react';
import { ArrowLeft, BookOpen, Loader2, List, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TextbookViewerProps {
    chapterId: number;
    title: string;
    onBack: () => void;
}

interface TocItem {
    id: string;
    text: string;
    level: number;
}

export function TextbookViewer({ chapterId, title, onBack }: TextbookViewerProps) {
    const [content, setContent] = useState<string>('');
    const [loading, setLoading] = useState(true);
    const [toc, setToc] = useState<TocItem[]>([]);
    const [showToc, setShowToc] = useState(false);
    const [scrollProgress, setScrollProgress] = useState(0);
    const [showScrollTop, setShowScrollTop] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setLoading(true);
        fetch(`/textbooks/${chapterId}.md`)
            .then(res => res.text())
            .then(text => {
                setContent(text);
                extractToc(text);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to load textbook:', err);
                setLoading(false);
            });
    }, [chapterId]);

    useEffect(() => {
        const handleScroll = () => {
            const winScroll = document.documentElement.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            setScrollProgress(scrolled);
            setShowScrollTop(winScroll > 300);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const extractToc = (text: string) => {
        const lines = text.split('\n');
        const items: TocItem[] = [];
        lines.forEach((line) => {
            if (line.startsWith('## ') || line.startsWith('### ')) {
                const text = line.replace(/^#+\s/, '');
                const id = text.replace(/\s+/g, '-').toLowerCase();
                items.push({
                    id,
                    text,
                    level: line.startsWith('## ') ? 2 : 3
                });
            }
        });
        setToc(items);
    };

    const scrollToSection = (id: string) => {
        const element = document.getElementById(id);
        if (element) {
            const offset = 100;
            const bodyRect = document.body.getBoundingClientRect().top;
            const elementRect = element.getBoundingClientRect().top;
            const elementPosition = elementRect - bodyRect;
            const offsetPosition = elementPosition - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
            setShowToc(false);
        }
    };

    const parseContent = (text: string) => {
        const lines = text.split('\n');
        const elements: React.ReactNode[] = [];
        let codeBlock: string[] = [];
        let isCodeBlock = false;
        let codeLanguage = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Code Block Start/End
            if (line.trim().startsWith('```')) {
                if (!isCodeBlock) {
                    isCodeBlock = true;
                    codeLanguage = line.trim().slice(3) || 'python';
                    codeBlock = [];
                } else {
                    isCodeBlock = false;
                    elements.push(
                        <div key={`code-${i}`} className="my-6 rounded-xl overflow-hidden border border-zinc-800 bg-[#0d0d0d] shadow-2xl">
                            <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/50 border-b border-zinc-800">
                                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{codeLanguage}</span>
                                <div className="flex gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                </div>
                            </div>
                            <pre className="p-4 overflow-x-auto text-sm leading-relaxed font-mono text-zinc-300">
                                <code>{codeBlock.join('\n')}</code>
                            </pre>
                        </div>
                    );
                }
                continue;
            }

            if (isCodeBlock) {
                codeBlock.push(line);
                continue;
            }

            // Horizontal Rule
            if (line.trim() === '---') {
                elements.push(<hr key={i} className="my-10 border-zinc-800" />);
                continue;
            }

            // Headers
            if (line.startsWith('# ')) {
                elements.push(<h1 key={i} className="text-3xl font-black text-white mt-12 mb-8 pb-4 border-b-2 border-indigo-500/20">{renderInline(line.slice(2))}</h1>);
                continue;
            }
            if (line.startsWith('## ')) {
                const headerText = line.slice(3);
                elements.push(
                    <h2
                        id={headerText.replace(/\s+/g, '-').toLowerCase()}
                        key={i}
                        className="text-2xl font-bold text-white mt-16 mb-6 flex items-center gap-3 scroll-mt-24"
                    >
                        <span className="w-1.5 h-6 bg-indigo-500 rounded-full" />
                        {renderInline(headerText)}
                    </h2>
                );
                continue;
            }
            if (line.startsWith('### ')) {
                const headerText = line.slice(4);
                elements.push(
                    <h3
                        id={headerText.replace(/\s+/g, '-').toLowerCase()}
                        key={i}
                        className="text-xl font-bold text-indigo-400 mt-10 mb-4 scroll-mt-24"
                    >
                        {renderInline(headerText)}
                    </h3>
                );
                continue;
            }

            // Numbered Lists
            if (/^\d+\.\s/.test(line)) {
                elements.push(
                    <div key={i} className="text-lg font-bold text-white mt-8 mb-4 flex gap-3 items-baseline">
                        <span className="text-indigo-500 tabular-nums">{line.split('.')[0]}.</span>
                        <div>{renderInline(line.replace(/^\d+\.\s/, ''))}</div>
                    </div>
                );
                continue;
            }

            // Bullets
            if (line.trim().startsWith('•') || line.trim().startsWith('- ')) {
                elements.push(
                    <div key={i} className="text-zinc-300 ml-4 mb-3 list-none flex items-start group">
                        <span className="text-indigo-500/50 mr-3 mt-2 w-1.5 h-1.5 bg-indigo-500 rounded-full shrink-0 group-hover:scale-125 transition-transform" />
                        <span className="leading-relaxed">{renderInline(line.replace(/^[•-]\s*/, ''))}</span>
                    </div>
                );
                continue;
            }
            if (line.trim().startsWith('◦')) {
                elements.push(
                    <div key={i} className="text-zinc-400 ml-10 mb-2 list-none text-sm flex items-start">
                        <span className="text-zinc-600 mr-2 mt-2 w-1 h-1 border border-zinc-600 rounded-full shrink-0" />
                        <span className="leading-relaxed">{renderInline(line.trim().slice(1))}</span>
                    </div>
                );
                continue;
            }

            // Empty lines
            if (!line.trim()) {
                elements.push(<div key={i} className="h-4" />);
                continue;
            }

            // Default Paragraph
            elements.push(<p key={i} className="text-zinc-300 leading-relaxed mb-4 text-justify">{renderInline(line)}</p>);
        }

        return elements;
    };

    const renderInline = (text: string) => {
        // First handle bold **text**
        let parts = text.split(/(\*\*.*?\*\*)/);
        let elements: React.ReactNode[] = parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={`bold-${i}`} className="text-white font-bold bg-zinc-800/50 px-1 rounded">{part.slice(2, -2)}</strong>;
            }
            return part;
        });

        // Then handle inline code `code`
        return elements.map((element, i) => {
            if (typeof element !== 'string') return element;

            const subParts = element.split(/(`.*?`)/);
            return subParts.map((subPart, j) => {
                if (subPart.startsWith('`') && subPart.endsWith('`')) {
                    return <code key={`code-${i}-${j}`} className="text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded font-mono text-[0.9em] border border-indigo-500/20">{subPart.slice(1, -1)}</code>;
                }
                return subPart;
            });
        });
    };

    if (loading) {
        return (
            <div className="flex h-[50vh] items-center justify-center">
                <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="relative animate-in fade-in duration-500">
            {/* Scroll Progress Bar */}
            <div className="fixed top-0 left-0 w-full h-1 z-[100] bg-zinc-900">
                <div
                    className="h-full bg-indigo-500 transition-all duration-150 ease-out"
                    style={{ width: `${scrollProgress}%` }}
                />
            </div>

            {/* Sticky Header */}
            <div className="fixed top-0 left-0 w-full z-50 bg-black/60 backdrop-blur-xl border-b border-white/5 py-4 px-6 md:px-12">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    <button
                        onClick={onBack}
                        className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors group text-sm md:text-base"
                    >
                        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                        <span className="hidden sm:inline">목록으로</span>
                    </button>

                    <h2 className="text-zinc-100 font-bold truncate max-w-[50%] md:max-w-none">
                        {title}
                    </h2>

                    <button
                        onClick={() => setShowToc(!showToc)}
                        className={cn(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all",
                            showToc
                                ? "bg-indigo-500 border-indigo-400 text-white"
                                : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700"
                        )}
                    >
                        <List className="w-4 h-4" />
                        <span className="text-xs font-medium hidden sm:inline">목차</span>
                    </button>
                </div>
            </div>

            {/* Table of Contents Overlay */}
            {showToc && (
                <div className="fixed inset-0 z-[60]">
                    <div
                        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                        onClick={() => setShowToc(false)}
                    />
                    <div className="absolute right-4 top-20 w-72 max-h-[70vh] overflow-y-auto bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl p-6 animate-in slide-in-from-right-4">
                        <div className="flex items-center gap-2 mb-6">
                            <List className="w-5 h-5 text-indigo-500" />
                            <h3 className="text-lg font-bold text-white">학습 목차</h3>
                        </div>
                        <div className="space-y-1">
                            {toc.map((item, i) => (
                                <button
                                    key={i}
                                    onClick={() => scrollToSection(item.id)}
                                    className={cn(
                                        "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
                                        item.level === 2
                                            ? "text-zinc-300 font-bold hover:bg-zinc-800 hover:text-white"
                                            : "text-zinc-500 pl-6 hover:bg-zinc-800 hover:text-white"
                                    )}
                                >
                                    {item.text}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <div className="h-24" /> {/* Header spacer */}

            <div className="max-w-4xl mx-auto px-6">
                <div className="glass-card bg-[#0a0a0a] border-zinc-800 p-8 md:p-12 shadow-2xl rounded-3xl">
                    <div className="prose prose-invert max-w-none">
                        {parseContent(content)}
                    </div>
                </div>
            </div>

            {/* Scroll to Top */}
            <button
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className={cn(
                    "fixed bottom-8 right-8 p-4 rounded-full bg-indigo-500 text-white shadow-2xl transition-all duration-300 hover:scale-110 active:scale-95 group z-[70]",
                    showScrollTop ? "translate-y-0 opacity-100" : "translate-y-20 opacity-0"
                )}
                aria-label="맨 위로 이동"
            >
                <ChevronUp className="w-6 h-6" />
            </button>

            <div className="h-32" /> {/* Bottom spacer */}
        </div>
    );
}
