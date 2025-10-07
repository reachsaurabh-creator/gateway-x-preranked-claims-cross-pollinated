// Debug script to test frontend functionality
const testData = {
    "run_id": "0efdff479729",
    "query": "What are the benefits of exercise?",
    "best_claim": "[openai] Exercise significantly improves cardiovascular health by strengthening the heart muscle, enhancing circulation, and reducing blood pressure, which collectively decreases the risk of heart disease and stroke. Regular physical activity also helps maintain healthy cholesterol levels by increasing high-density lipoprotein (HDL) and lowering triglycerides, contributing to overall heart health.",
    "confidence": 0.5,
    "rounds": 3,
    "total_duels": 3,
    "stop_reason": "budget_exhausted"
};

const testTimeline = [
    {
        "run_id": "0efdff479729",
        "round_index": 1,
        "convergence_score": 0.164,
        "best_claim_cid": "[openai] Exercise significantly improves cardiovascular health by strengthening the heart muscle, enhancing circulation, and reducing blood pressure, which collectively decreases the risk of heart disease and stroke. Regular physical activity also helps maintain healthy cholesterol levels by increasing high-density lipoprotein (HDL) and lowering triglycerides, contributing to overall heart health.",
        "best_claim_text": "[openai] Exercise significantly improves cardiovascular health by strengthening the heart muscle, enhancing circulation, and reducing blood pressure, which collectively decreases the risk of heart disease and stroke. Regular physical activity also helps maintain healthy cholesterol levels by increasing high-density lipoprotein (HDL) and lowering triglycerides, contributing to overall heart health.",
        "summary": "Truth response selects [openai] Exercise significantly improves cardiovascular health by strengthening the heart muscle, enhancing circulation, and reducing blood pressure, which collectively decreases the risk of heart disease and stroke. Regular physical activity also helps maintain healthy cholesterol levels by increasing high-density lipoprotein (HDL) and lowering triglycerides, contributing to overall heart health. as current best. Next contenders: [xai] Regular exercise provides numerous physical and mental health benefits, including improved cardiovascular function, enhanced muscle strength and endurance, better bone density, weight management, and reduced risk of chronic diseases such as diabetes, heart disease, and certain cancers. Additionally, exercise releases endorphins that boost mood, reduce stress and anxiety, improve sleep quality, and enhance cognitive function, making it a comprehensive approach to overall well-being. (TS 0.25), [gemini] Exercise offers a wide range of physical and mental health benefits. Physically, it strengthens the cardiovascular system, builds muscle and bone density, improves flexibility and balance, and helps maintain a healthy weight. Mentally, regular physical activity reduces stress, anxiety, and depression while boosting mood and cognitive function. Additionally, exercise can improve sleep quality, increase energy levels, and enhance overall quality of life. (TS 0.15). Some divergence remains (CI overlap with best).",
        "top_claims": [
            {
                "cid": "[openai] Exercise significantly improves cardiovascular health by strengthening the heart muscle, enhancing circulation, and reducing blood pressure, which collectively decreases the risk of heart disease and stroke. Regular physical activity also helps maintain healthy cholesterol levels by increasing high-density lipoprotein (HDL) and lowering triglycerides, contributing to overall heart health.",
                "score": 0.6,
                "ci_low": 0.4,
                "ci_high": 0.8
            },
            {
                "cid": "[xai] Regular exercise provides numerous physical and mental health benefits, including improved cardiovascular function, enhanced muscle strength and endurance, better bone density, weight management, and reduced risk of chronic diseases such as diabetes, heart disease, and certain cancers. Additionally, exercise releases endorphins that boost mood, reduce stress and anxiety, improve sleep quality, and enhance cognitive function, making it a comprehensive approach to overall well-being.",
                "score": 0.25,
                "ci_low": 0.1,
                "ci_high": 0.4
            },
            {
                "cid": "[gemini] Exercise offers a wide range of physical and mental health benefits. Physically, it strengthens the cardiovascular system, builds muscle and bone density, improves flexibility and balance, and helps maintain a healthy weight. Mentally, regular physical activity reduces stress, anxiety, and depression while boosting mood and cognitive function. Additionally, exercise can improve sleep quality, increase energy levels, and enhance overall quality of life.",
                "score": 0.15,
                "ci_low": 0.05,
                "ci_high": 0.25
            }
        ]
    }
];

// Test convergence chart data extraction
function testConvergenceChart(timeline) {
    console.log('Testing convergence chart...');
    const scores = timeline.map(round => round.convergence_score);
    console.log('Convergence scores:', scores);
    console.log('Max score:', Math.max(...scores, 1.0));
    console.log('Min score:', Math.min(...scores, 0.0));
    return scores.length > 0;
}

// Test round details display
function testRoundDetails(timeline) {
    console.log('Testing round details...');
    let allValid = true;
    timeline.forEach((round, index) => {
        console.log(`\nRound ${index + 1}:`);
        console.log('  Convergence score:', round.convergence_score);
        console.log('  Best claim text length:', round.best_claim_text?.length || 0);
        console.log('  Summary length:', round.summary?.length || 0);
        console.log('  Top claims count:', round.top_claims?.length || 0);
        
        const hasBestClaim = round.best_claim_text && round.best_claim_text.length > 0;
        const hasSummary = round.summary && round.summary.length > 0;
        const hasTopClaims = round.top_claims && round.top_claims.length > 0;
        
        console.log('  Has best claim:', hasBestClaim);
        console.log('  Has summary:', hasSummary);
        console.log('  Has top claims:', hasTopClaims);
        
        if (!hasBestClaim || !hasSummary || !hasTopClaims) {
            allValid = false;
        }
    });
    return allValid;
}

// Run tests
console.log('=== Frontend Debug Test ===');
console.log('Test data:', testData);
console.log('Test timeline length:', testTimeline.length);

const chartTest = testConvergenceChart(testTimeline);
const detailsTest = testRoundDetails(testTimeline);

console.log('\n=== Results ===');
console.log('Convergence Chart Test:', chartTest ? '✅ PASS' : '❌ FAIL');
console.log('Round Details Test:', detailsTest ? '✅ PASS' : '❌ FAIL');

