const SEMESTER_SLUG = /^suda-se-guide\/大([一二三四])([上下])$/;

/** @param {Array<{ slug: string, data: { draft?: boolean } }>} posts */
export function getSemesterGuides(posts) {
	return posts
		.flatMap((post) => {
			const match = SEMESTER_SLUG.exec(post.slug);
			if (!match || post.data.draft === true) return [];

			const semester = `大${match[1]}${match[2]}`;
			return [{
				slug: post.slug,
				semester,
				order: "一二三四".indexOf(match[1]) * 2 + "上下".indexOf(match[2]),
			}];
		})
		.sort((a, b) => a.order - b.order)
		.map(({ slug, semester }) => ({ slug, semester }));
}
