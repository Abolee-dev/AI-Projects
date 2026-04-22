from collections import Counter, defaultdict
from typing import List, Dict


class TrendAnalyzer:
    def extract_trends(self, records: List[Dict]) -> Dict:
        topic_by_year = defaultdict(list)

        for r in records:
            year = r.get("publication_year")
            for topic in r.get("topic_tags", []):
                topic_by_year[year].append(topic)

        trends = {}
        for year, topics in topic_by_year.items():
            trends[year] = Counter(topics).most_common(10)

        return trends
