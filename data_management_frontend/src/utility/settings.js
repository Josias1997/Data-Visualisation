export const options = {
        sorting: true,
        grouping: true,
        selection: true,
        filtering: true,
        exportButton: true,
    };

export const localization = {
    pagination: {
        labelDisplayedRows: '{from}-{to} sur {count}',
        labelRowsSelect: 'lignes',
        labelRowsPerPage: 'Lignes par page',
        firstAriaLabel: 'Première Page',
        firstTooltip: 'Première Page',
        previousAriaLabel: 'Page précédente',
        previousTooltip: 'Page précédente',
        nextAriaLabel: 'Page suivante',
        nextTooltip: 'Page suivante',
        lastAriaLabel: 'Dernière Page',
        lastTooltip: 'Dernière Page',
    },
    toolbar: {
        nRowsSelected: '{0} ligne(s) sélectionnée(s)',
        addRemoveColumns: 'Ajouter ou Supprimer des colonnes',
        showColumnsTitle: 'Afficher les colonnes',
        showColumnsAriaLabel: 'Afficher les colonnes',
        exportTitle: 'Exporter',
        exportAriaLabel: 'Exporter',
        exportName: 'Exporter sous forme CSV',
        searchTooltip: 'Rechercher',
        searchPlaceholder: 'Rechercher',
    },
    header: {
        actions: 'Actions',
    },
    body: {
        emptyDataSourceMessage: 'Aucun résultat à afficher',
        filterRow: {
            filterTooltip: 'Filtrer',
        },
        addTooltip: 'Ajouter',
        deleteTooltip: 'Supprimer',
        editTooltip: 'Modifier',
        editRow: {
            deleteText: 'Êtes vous sûr de vouloir supprimer cette ligne ?',
            cancelTooltip: 'Annuler',
            saveTooltip: 'Enregistrer'
        },
        grouping: {
            placeholder: 'Déplacez les entêtes pour faire un group by'
        }
    }
};