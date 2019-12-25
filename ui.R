#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(shinyjs)
library(shinyalert)
library(dashboardthemes)

# Define UI for application that draws a histogram
shinyUI(
    
    dashboardPage(
        
        
        dashboardHeader(title = "Clustering"),
        sidebar <- dashboardSidebar(
            sidebarMenu(id = 'sidebarmenu',
                        menuItem('Introduction', tabName = 'Introduction', icon = icon('align-justify')),
                        menuItem('Different Methods', tabName = 'Different_Methods',
                                 icon = icon('angle-double-down'),
                                 menuItem('Parametric Clustering',
                                          tabName = 'para_clust',
                                          icon = icon('angle-double-down')),
                                 menuItem('Non-Parametric Clustering',
                                          tabName = 'nonparam_clust',
                                          icon = icon('angle-double-down'),
                                          menuSubItem('Distance Based',
                                                   tabName = 'distance',
                                                   icon = icon('angle-double-down')),
                                          menuSubItem('Density Based',
                                                   tabName = 'density',
                                                   icon = icon('angle-double-down')))),
                        menuItem('Cluster Validation', tabName = 'Validation',
                                 icon = icon('angle-double-down')),
                        menuItem('Conclusion and References', tabName = 'end',
                                 icon = icon('angle-double-down')))),
        ## https://github.com/nik01010/dashboardthemes   dashboardthemes package is used for custom theme on the default shiny dashboard        
        
        dashboardBody(
            ### changing theme
            shinyDashboardThemes(
                theme = "blue_gradient"
            ),
            
            tabItems(
                tabItem("Introduction",
                        uiOutput("intro_text"),
                        imageOutput("Intro", height = "auto"),
                        uiOutput('intro_text1')),
                tabItem("para_clust",
                        uiOutput("intro_text2"),
                        uiOutput('formula'),
                        uiOutput('formula1'),
                        uiOutput("intro_text3"),
                        uiOutput('formula2'),
                        uiOutput('formula3'),
                        uiOutput("intro_text4"),
                        uiOutput('formula4'),
                        uiOutput("intro_text5"),
                        uiOutput('formula5'),
                        plotOutput('gmm')),
                tabItem("density",
                        uiOutput("intro_text9"),
                        
                        imageOutput("db2", height = "auto"),
                        uiOutput('intro_text10'),
                        imageOutput("db1", height = "auto"),
                        uiOutput('intro_text20'),
                        
                        fluidRow(column(width = 4,
                                        sliderInput("epsilon", "Radius of the Neighborhood: ", min = 0.01, max = 2, value = 1),
                                        sliderInput("n", "Minimum number of neighbor points ", min = 1, max = 10, value = 4)),
                                 plotOutput("DBSCAN"))),
                tabItem("distance",
                        uiOutput('intro_text11'),
                        uiOutput('formula6'),
                        uiOutput('intro_text12'),
                        textOutput("methods_text"),
                        sliderInput("k", "Number of clusters: ", min = 1, max = 10, value = 4),
                        plotOutput("cluster_plot_kmeans"),
                        uiOutput('intro_text13'),
                        sliderInput("k1", "Number of clusters: ", min = 1, max = 10, value = 4),
                        plotOutput("cluster_plot_hierdend"),
                        plotOutput("cluster_plot_hier"),
                        uiOutput('intro_text14'),
                        imageOutput("link", height = "auto")),
                tabItem("Validation",
                        uiOutput('intro_test14'),
                        uiOutput('formula7')),
                tabItem("end",
                        uiOutput('intro_test19'),
                        uiOutput('intro_test18')
            )
           
        )
        
    )
))

