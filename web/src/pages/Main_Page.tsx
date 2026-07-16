import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react';
import '../index.css'

interface DYK_OTD_Data {
    content: string
}
interface ArticleData {
    title: string;
    abstract: string;
}

function Main_Page() {
    const [dyk, setDyk] = useState<DYK_OTD_Data[]>([]);
    const [article, setArticle] = useState<ArticleData[]>([]);
    const [popular, setPopular] = useState<ArticleData[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const pickDYK = <T,>(arr: T[], seed: number, n = 10): NonNullable<T>[] => {
            let s = seed >>> 0;
            const rand = () => (s = (s * 1664525 + 1013904223) & 0xFFFFFFFF) / 2 ** 32;
            const copy = [...arr];

            for (let i = copy.length - 1; i > 0; i--) {
                const j = Math.floor(rand() * (i + 1));
                [copy[i], copy[j]] = [copy[j], copy[i]];
            }

            const result: NonNullable<T>[] = [];
            for (let i = 0; i < copy.length && result.length < n; i++) {
                if (copy[i] != null) {
                    result.push(copy[i] as NonNullable<T>);
                }
            }
            return result;
        };

        const pickArticle = (arr: any[], seed: number) => {
            if (!arr || arr.length === 0) {
                return { title: '', abstract: '' };
            }

            const len = typeof arr.length === 'number' ? arr.length : 0;
            if (len === 0) {
                return { title: '', abstract: '' };
            }

            let s = seed >>> 0;
            const rand = () => (s = (s * 1664525 + 1013904223) & 0xFFFFFFFF) / 2 ** 32;

            const index = Math.floor(rand() * len) % len;
            let picked = arr[index];

            if (!picked) {
                const fallback = arr.find(item => item != null);
                if (fallback) {
                    return {
                        title: fallback.title ?? '',
                        abstract: fallback.sections?.Abstract ?? '',
                    };
                }
                return { title: '', abstract: '' };
            }

            return {
                title: picked.title ?? '',
                abstract: picked.sections?.Abstract ?? '',
            };
        };

        const dateSeed = +new Date().toISOString().slice(0, 10).replace(/-/g, ''); // yyyymmdd

        const fetchDYK = async () => {
            try {
                const response = await fetch('/dyk.json');
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let data: DYK_OTD_Data[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        data = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        data = [parsed as DYK_OTD_Data];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch {
                    console.log("Error parsing DYK")
                }

                const selected = pickDYK(data, dateSeed, 10);
                setDyk(selected);
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        const fetchArticles = async () => {
            try {
                const response = await fetch('/articles.json');
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let data: ArticleData[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        data = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        data = [parsed as ArticleData];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch (err) {
                    setError(err instanceof Error ? err.message : String(err));
                } finally {
                    setLoading(false);
                }

                const selected = pickArticle(data, dateSeed)
                setArticle([selected]);
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        const fetchPopular = async () => {
            try {
                const response = await fetch('/articles.json');
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let data: ArticleData[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        data = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        data = [parsed as ArticleData];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch {
                    data = text
                        .split('\n')
                        .filter(line => line.trim() !== '')
                        .map(line => JSON.parse(line) as ArticleData);
                }

                setPopular(data.slice(0, 16));
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        fetchDYK();
        fetchArticles();
        fetchPopular();
    }, []);

    if (loading) return <div>Loading article list...</div>;
    if (error) return <div style={{ color: 'red' }}>Error: {error}</div>;

    return (
        <>
            <div className="main-container">
                <img id="main-img" src="/fake2.png" alt="logo" />
                <section id="welcome">
                    <h1>Welcome to <a href="https://github.com/mzums/fakewiki">FakeWiki</a>,</h1>
                    <h3>the fake encyclopedia maintained by <a href="https://mzums.com">mzums</a></h3>
                    <h3>
                        Articles generated by a custom-trained AI model based on data from
                        <a href="https://en.wikipedia.org"> Wikipedia</a>.
                    </h3>
                </section>
            </div>



            <section id="cards">
                <h4 id="view-all">
                    view  <Link to="/All_Articles">all articles...</Link>
                </h4>
                <section id="card-row">
                    <section id="card">
                        <section id="card-title">
                            From today's featured article
                        </section>
                        {article.map((item, index) => (
                            <p key={index} id="card-text">{item.abstract}</p>
                        ))}
                    </section>

                    <section id="card">
                        <section id="card-title">
                            Most popular
                        </section>
                        <ul style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px' }}>
                            {popular.map((article, _) => (
                                <li>
                                    <Link to={`/Article/${article.title}`}>
                                        {article.title}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </section>
                </section>

                <section id="card-row">
                    <section id="card">
                        <section id="card-title">
                            Did you know
                        </section>
                        <ul>
                            {dyk.map((dyk, index) => (
                                <li key={index}>
                                    {dyk.content}
                                </li>
                            ))}
                        </ul>
                    </section>

                </section>
            </section >
        </>
    )
}

export default Main_Page