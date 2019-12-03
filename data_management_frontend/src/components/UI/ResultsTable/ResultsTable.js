import React from 'react';
import Table from './Table/Table';

const ResultsTable = ({test, result}) => {
    let data = {
        columns: [
            {
                label: 'Fonction',
                field: 'function',
            },
            {
                label: 'Statistic',
                field: 'statistic'
            },
            {
                label: 'Pvalue',
                field: 'pvalue'
            }
        ],
        rows: [
            {
                'function': 'normaltest',
                'statistic': result[0],
                'pvalue': result[1]
            }
        ]
    };
    switch(test) {
        case 'normtest':
            if(result.normaltest !== undefined) {
                const {normaltest, shapiro} = result;
                data = {
                    columns: [
                        {
                            label: 'Fonctions',
                            field: 'function',
                        },
                        {
                            label: 'Statistic',
                            field: 'statistic'
                        },
                        {
                            label: 'Pvalue',
                            field: 'pvalue'
                        }
                    ],
                    rows: [
                        {
                            'function': 'normaltest',
                            'statistic': normaltest[0],
                            'pvalue': normaltest[1]
                        },
                        {
                            'function': 'shapiro',
                            'statistic': shapiro[0],
                            'pvalue': shapiro[1]
                        }
                    ]
                };
            }
            break;
        case 'skewtest':
            data.rows[0].function = 'skewtest';
            break;
        case 'cumfreq':
            if (result.cumcount !== undefined) {
                const {cumcount, lowerlimit, binsize, extrapoints} = result;
                data = {
                    columns: [
                        {
                            label: 'Fonction',
                            field: 'function'
                        },
                        {
                            label: 'Cumcount',
                            field: 'cumcount'
                        },
                        {
                            label: 'Lowerlimint',
                            field: 'lowerlimit'
                        },
                        {
                            label: 'Binsize',
                            field: 'binsize'
                        },
                        {
                            label: 'Extrapoints',
                            field: 'extrapoints'
                        }
                    ],
                    rows: [
                        {
                            'function': "cumfreq",
                            'cumcount': cumcount.map(freq => freq + " "),
                            'lowerlimit': lowerlimit,
                            'binsize': binsize,
                            'extrapoints': extrapoints
                        }
                    ]
                }
            }
            break;
        case 'correlation':
            if(result.pearsonr !== undefined) {
                const { pearsonr, spearmanr } = result;
                data = {
                    columns: [
                        {
                            label: 'Fonction',
                            field: 'function',
                        },
                        {
                            label: 'Correlation',
                            field: 'correlation'
                        },
                        {
                            label: 'Pvalue',
                            field: 'pvalue'
                        }
                    ],
                    rows: [
                        {
                            'function': 'pearsonr',
                            'correlation': pearsonr[0],
                            'pvalue': pearsonr[1]
                        },
                        {
                            'function': 'spearmanr',
                            'correlation': spearmanr[0],
                            'pvalue': spearmanr[1]
                        }
                    ]
                };
            }
            break;
        case 't-test':
            data.rows[0].function = 'ttest_ind';
            break;
        case 'anova':
            data.rows[0].function = 'f_oneway';
            break;
        case 'chisquare':
                data = {
                    columns: [
                        {
                            label: 'Fonction',
                            field: 'function',
                        },
                        {
                            label: 'Chisq',
                            field: 'chisq'
                        },
                        {
                            label: 'P',
                            field: 'p'
                        }
                    ],
                    rows: [
                        {
                            'function': 'chisquare',
                            'chisq': result[0],
                            'p': result[1]
                        }
                    ]
                };
            break;
        case 'fisher_exact':
            data = {
                columns: [
                    {
                        label: 'Fonction',
                        field: 'function',
                    },
                    {
                        label: 'Oddsratio',
                        field: 'oddsratio'
                    },
                    {
                        label: 'Pvalue',
                        field: 'pvalue'
                    }
                ],
                rows: [
                    {
                        'function': 'fisher_exact',
                        'oddsratio': result[0],
                        'pvalue': result[1]
                    }
                ]
            };
            break;
        case 'wilcoxon':
            data.rows[0].function = 'wilcoxon';
            break;
        case 'zscore':
            if (Array.isArray(result)) {
                data = {
                    columns: [
                        {
                            label: 'Fonction',
                            field: 'function',
                        },
                        {
                            label: 'Values',
                            field: 'values'
                        }
                    ],
                    rows: result.map(value => {
                            return {
                                'function': 'zscore',
                                'values': value,
                            };
                        })
        
                };
            }
            break;
    }
    return (
        <div className="mt-5">
            <Table data={data} />
        </div>
    );
};

export default ResultsTable;