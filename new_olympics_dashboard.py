<<<<<<< HEAD
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="The Olympic Story: 120 Years of History",
    layout="wide",
    page_icon="🏅"
)


    # 🧾 Your short intro
with st.sidebar.expander("👤 About Hassan"):
    st.image("hassan.png", width=100)
    st.markdown("""
    **Hassan Raza**  
    *Data Analyst*  
    Passionate about uncovering stories hidden in data.  
    
    📫 **Email:** razakhattak123@gmail.com  
    
    📊 **Whatsapp:**  +92335 5102051
    
    🔗 **LinkedIn:** [https://www.linkedin.com/in/hassan-raza-9651b6279)
    
    🔗 **Kaggle:** [https://www.kaggle.com/hassanrazakhattak)
    """)

# 🏅 Title and subtitle on the right
st.title("The Olympic Story: 120 Years of History")
st.markdown("*A deep dive into the athletes, medals, and nations that shaped Olympic history.*")


# 📚 About Dataset in Sidebar
with st.sidebar.expander("📚 About Dataset", expanded=False):
    st.markdown("""
**Olympic History Dataset (1896–2016)**  
_Source: [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)_

This dataset provides a detailed record of Olympic Games history across **both Summer and Winter Games**.

**Key Features:**
- 👤 **Athlete Info**: Name, gender, age, height, weight  
- 🏟️ **Event Info**: Year, season, city, sport, and event  
- 🌍 **Country Details**: Team and NOC codes  
- 🥇 **Medal Records**: Gold, Silver, Bronze (or none)  

**Scale:**
- Over **270,000 athlete entries**
- Covers **120 years** of Olympic history

This rich dataset allows you to explore long-term trends in global athletic participation and medal performance.
""")


##----------------------------- Section 2 FIlters--------------------------------


# Load dataset
df = pd.read_csv("athlete_events.csv")

# Sidebar filters
st.sidebar.header("🎛️ Filter Olympic Data")

# Country filter
country_list = ['All'] + sorted(df['NOC'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country (NOC)", country_list)

# Medal filter
medal_list = ['All', 'Gold', 'Silver', 'Bronze']
selected_medal = st.sidebar.selectbox("Select Medal Type", medal_list)

# Year filter
year_list = ['All'] + sorted(df['Year'].dropna().unique().astype(int).tolist())
selected_year = st.sidebar.selectbox("Select Year", year_list)

# Apply filters
filtered_df = df.copy()

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['NOC'] == selected_country]

if selected_medal != 'All':
    filtered_df = filtered_df[filtered_df['Medal'] == selected_medal]

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == int(selected_year)]



##----------------------------- SECTION 3: Key Metrics Summary (KPIs)--------------------------------


st.markdown("### 📊 Olympic Games Summary")

# Calculate key metrics
total_athletes = filtered_df['ID'].nunique()
total_medals = filtered_df['Medal'].notnull().sum()
total_countries = filtered_df['NOC'].nunique()
total_years = filtered_df['Year'].nunique()

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Athletes", f"{total_athletes:,}")
col2.metric("🥇 Total Medals Awarded", f"{total_medals:,}")
col3.metric("🌍 Participating Countries", f"{total_countries}")
col4.metric("📅 Olympic Years", f"{total_years}")


# ----------------------------- SECTION 4: Visual Insights -----------------------------

st.markdown("### Visual Insights: Olympic Medals Summary")

# Filter medal winners only
medal_winners = filtered_df[filtered_df['Medal'].notnull()]

# Create 3 equal columns for charts
col1, col2, col3 = st.columns(3)

if selected_country == 'All':
    # Top 10 medal-winning countries — shown in col1
    with col1:
        st.markdown("🏅 Top 10 Medal-Winning Countries")

        top_countries = (
            medal_winners.groupby('NOC')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = top_countries.plot(kind='barh', color='gold', ax=ax)

        ax.set_xlabel("Total Medals", fontsize=9)
        ax.set_ylabel("Country (NOC)", fontsize=9)
        ax.set_title("", fontsize=0)
        ax.invert_yaxis()  # So top country shows on top

        # Add value labels
        for p in bars.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{int(width)}',
                            (width + 1, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=9, weight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    # Specific country breakdown — shown in col1
    with col1:
        st.markdown(f"🏅 Medal Breakdown for {selected_country}")

        country_medals = (
            medal_winners[medal_winners['NOC'] == selected_country]
            .groupby('Medal')
            .size()
            .reindex(['Gold', 'Silver', 'Bronze'], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32']
        bars = country_medals.plot(kind='barh', color=colors, ax=ax)

        ax.set_xlabel("Number of Medals", fontsize=9)
        ax.set_ylabel("")
        ax.set_title("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for p in bars.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{int(width)}',
                            (width + 0.5, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=9, weight='bold')

        plt.tight_layout()
        st.pyplot(fig)


# col2-------------------------------

with col2:
    st.markdown("📈 Athlete Participation Over Time")

    # Filter data for selected country (or all)
    if selected_country != 'All':
        country_df = filtered_df[filtered_df['NOC'] == selected_country]
    else:
        country_df = filtered_df.copy()

    # Group by Year and Gender
    athlete_counts = (
        country_df.drop_duplicates(subset=['ID', 'Year'])  # Avoid duplicate athlete counts
        .groupby(['Year', 'Sex'])
        .size()
        .unstack(fill_value=0)
        .rename(columns={'M': 'Male', 'F': 'Female'})
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    athlete_counts.plot(ax=ax, marker='o', linewidth=2)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Number of Athletes", fontsize=9)

    title_label = f"Athletes from {selected_country}" if selected_country != 'All' else "All Athletes"
    ax.set_title(title_label, fontsize=10, fontweight='bold')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title="Gender", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)


 #col3-----------------------------
 
with col3:
    st.markdown("🧑‍🤝‍🧑 Gender Distribution")

    if selected_country != 'All':
        gender_df = filtered_df[filtered_df['NOC'] == selected_country]
    else:
        gender_df = filtered_df.copy()

    gender_counts = (
        gender_df.drop_duplicates(subset=['ID', 'Year'])  # Avoid double counts
        .groupby('Sex')
        .size()
        .reindex(['M', 'F'], fill_value=0)
        .rename(index={'M': 'Male', 'F': 'Female'})
    )

    fig, ax = plt.subplots(figsize=(5, 2.6))
    colors = ['skyblue', 'lightcoral']
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors, textprops={'fontsize': 9})
    ax.axis('equal')
    ax.set_title(f"Gender Ratio ({selected_country})", fontsize=9)
    st.pyplot(fig)


# ----------------------------- SECTION 5: Top Medalists by Gender -----------------------------

st.markdown("### 🧑‍🤝‍🧑 Top Medalists by Gender")

# Confirm gender values in dataset (case-sensitive)
gender_choice = st.selectbox("Select Gender", ['M', 'F'])

# Filter: only medal winners of selected gender
gender_data = filtered_df[
    (filtered_df['Sex'] == gender_choice) &
    (filtered_df['Medal'].notnull())
]

# Apply country filter
if selected_country != 'All':
    gender_data = gender_data[gender_data['NOC'] == selected_country]

# Group by athlete name and country
top_gender_medalists = (
    gender_data.groupby(['Name', 'NOC'])
    .size()
    .reset_index(name='Medal Count')
    .sort_values(by='Medal Count', ascending=False)
    .head(10)
)

# Create label: Name (NOC)
top_gender_medalists['Athlete'] = top_gender_medalists['Name'] + " (" + top_gender_medalists['NOC'] + ")"

# Plot with Plotly
import plotly.express as px

if not top_gender_medalists.empty:
    fig = px.bar(
        top_gender_medalists,
        x='Medal Count',
        y='Athlete',
        orientation='h',
        text='Medal Count',
        color='Medal Count',
        color_continuous_scale='Blues' if gender_choice == 'M' else 'Pinkyl',
        title=f"Top 10 {'Male' if gender_choice == 'M' else 'Female'} Medalists in {selected_country if selected_country != 'All' else 'All Countries'}"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Number of Medals",
        yaxis_title="Athlete (Country)",
        height=450,
        margin=dict(t=50, l=100, r=20, b=40)
    )

    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No {'Male' if gender_choice == 'M' else 'Female'} medalists found for {selected_country}")


# ----------------------------- SECTION 6: Sport Dominance with Country Hover -----------------------------

st.markdown("### 🌍 Sports by Country Dominance")

# Filter medal winners only
sports_df = filtered_df[filtered_df['Medal'].notnull()]

# If specific country selected, keep it filtered
if selected_country != 'All':
    sports_df = sports_df[sports_df['NOC'] == selected_country]

# Group by Sport and Country (NOC)
sport_country = (
    sports_df.groupby(['Sport', 'NOC'])
    .size()
    .reset_index(name='Medal Count')
    .sort_values(by='Medal Count', ascending=False)
)

import plotly.express as px

if not sport_country.empty:
    fig = px.treemap(
        sport_country,
        path=['Sport', 'NOC'],
        values='Medal Count',
        color='Medal Count',
        color_continuous_scale='thermal',
        title=f"🥇 Most Dominant Countries by Sport ({selected_country if selected_country != 'All' else 'All Countries'})",
        hover_data={'Medal Count': True, 'Sport': True, 'NOC': True}
    )

    fig.update_layout(
        margin=dict(t=40, l=20, r=20, b=20),
        height=500
    )

    # 🎯 Center using Streamlit layout
    left, center, right = st.columns([1, 4, 1])

    with center:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No data available for selected filters.")


# ----------------------------- SECTION 7: Athlete Records & Extremes -----------------------------

st.markdown("### 🧍‍♂️ Athlete Profile Highlights")

# ➕ Add Gender Filter
gender_options = ['All'] + list(filtered_df['Sex'].dropna().unique())
selected_sex = st.selectbox("Filter by Gender", gender_options, index=0)

# 🔍 Apply Gender Filter
gender_df = filtered_df.copy()
if selected_sex != 'All':
    gender_df = gender_df[gender_df['Sex'] == selected_sex]

# Clean and deduplicate
valid_df = gender_df.drop_duplicates(subset=["ID", "Name", "Sex", "Age", "Height", "Weight", "Team", "NOC", "Event", "Medal"])
valid_df = valid_df.dropna(subset=["Age", "Height", "Weight"])

# Format text with compact styling
def format_athlete_row(row):
    medal = row["Medal"] if pd.notna(row["Medal"]) else "No Medal"
    return f"""
    <div style="font-size:13px; line-height:1.4">
    <strong>{row['Name']}</strong><br>
    <b>Country:</b> {row['NOC']}<br>
    <b>Event:</b> <i>{row['Event']}</i><br>
    <b>Age:</b> {int(row['Age'])} | <b>Height:</b> {int(row['Height'])} cm | <b>Weight:</b> {int(row['Weight'])} kg<br>
    <b>Medal:</b> {medal}
    </div>
    """

# Find records
if not valid_df.empty:
    heaviest = valid_df.loc[valid_df['Weight'].idxmax()]
    tallest = valid_df.loc[valid_df['Height'].idxmax()]
    shortest = valid_df.loc[valid_df['Height'].idxmin()]
    youngest = valid_df.loc[valid_df['Age'].idxmin()]
    oldest = valid_df.loc[valid_df['Age'].idxmax()]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("#### 🏋️ Heaviest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(heaviest), unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📏 Tallest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(tallest), unsafe_allow_html=True)

    with col3:
        st.markdown("#### 🧍‍♂️ Shortest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(shortest), unsafe_allow_html=True)

    with col4:
        st.markdown("#### 👶 Youngest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(youngest), unsafe_allow_html=True)

    with col5:
        st.markdown("#### 👴 Oldest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(oldest), unsafe_allow_html=True)
else:
    st.warning("No athletes found for the selected gender.")



# ----------------------------- SECTION 8: Most Successful Athletes -----------------------------

st.markdown("### 🏆 Most Successful Athletes")

# 👤 Gender Filter (with unique key to avoid duplicate ID error)
sex_options = ['All'] + list(filtered_df['Sex'].dropna().unique())
selected_sex = st.selectbox("Filter by Gender", sex_options, key="gender_filter_section8")

# 🥇 Step 1: Filter only medal winners
top_athletes_df = filtered_df[filtered_df["Medal"].notnull()].copy()

# 🧍‍♂️ Step 2: Apply gender filter
if selected_sex != 'All':
    top_athletes_df = top_athletes_df[top_athletes_df['Sex'] == selected_sex]

# 🧮 Step 3: Count medals per athlete
athlete_medal_summary = (
    top_athletes_df.groupby(["Name", "NOC"])
    .Medal.value_counts()
    .unstack(fill_value=0)
    .assign(Total=lambda df: df.sum(axis=1))
    .sort_values(by="Total", ascending=False)
    .reset_index()
)

# 📊 Step 4: Pick top 3
top3_athletes = athlete_medal_summary.head(3)

# 📝 Step 5: Card formatting utility
def format_athlete_card(row):
    return f"""
    <div style="font-size:14px; line-height:1.5; padding: 6px 10px; border-radius: 10px; ">
        <strong style="font-size:16px;">🏅 {row['Name']}</strong><br>
        <b>Country:</b> {row['NOC']}<br>
        <b>Gold:</b> 🥇 {row.get('Gold', 0)} |
        <b>Silver:</b> 🥈 {row.get('Silver', 0)} |
        <b>Bronze:</b> 🥉 {row.get('Bronze', 0)}<br>
        <b>Total:</b> <span style="color:green; font-weight:bold;">{row['Total']}</span>
    </div>
    """

# 🧱 Step 6: Layout — 3 equal columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🥇 1st", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[0]), unsafe_allow_html=True)

with col2:
    st.markdown("#### 🥈 2nd", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[1]), unsafe_allow_html=True)

with col3:
    st.markdown("#### 🥉 3rd", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[2]), unsafe_allow_html=True)



# ---------------------------- SECTION 9: SOUTH ASIA ANALYSIS ----------------------------


st.markdown("### 🌏 South Asia Olympic Analysis")
st.markdown("""
This section explores the Olympic journey of **South Asian countries**:  
🇮🇳 India, 🇵🇰 Pakistan, 🇧🇩 Bangladesh, 🇳🇵 Nepal, 🇱🇰 Sri Lanka, 🇧🇹 Bhutan, 🇲🇻 Maldives  
""")

# Country Codes and Labels
south_asia_nocs = ['IND', 'PAK', 'BAN', 'NEP', 'SRI', 'BHU', 'MDV']
country_labels = {
    'IND': 'India 🇮🇳', 'PAK': 'Pakistan 🇵🇰', 'BAN': 'Bangladesh 🇧🇩',
    'NEP': 'Nepal 🇳🇵', 'SRI': 'Sri Lanka 🇱🇰', 'BHU': 'Bhutan 🇧🇹', 'MDV': 'Maldives 🇲🇻'
}

sa_df = df[df['NOC'].isin(south_asia_nocs)]

# -------- A) Year-wise Participation --------
st.markdown("### 👟 Year-wise Athlete Participation")

participation = sa_df.groupby(['Year', 'NOC'])['ID'].nunique().reset_index(name='Athletes')
fig1 = px.line(participation, x='Year', y='Athletes', color='NOC',
               markers=True, template='plotly_white',
               labels={'NOC': 'Country'}, height=350)
fig1.for_each_trace(lambda t: t.update(name=country_labels.get(t.name, t.name)))
st.plotly_chart(fig1, use_container_width=True)

# -------- B) Total Medal Breakdown --------
st.markdown("### 🥇 Total Medals by Country")

medals_df = sa_df[sa_df['Medal'].notnull()]
medal_summary = medals_df.groupby(['NOC', 'Medal']).size().reset_index(name='Count')

fig2 = px.bar(medal_summary, x='NOC', y='Count', color='Medal',
              barmode='group',
              color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'},
              labels={'NOC': 'Country'}, height=350, template='plotly_white')
fig2.update_xaxes(ticktext=[country_labels[noc] for noc in south_asia_nocs], tickvals=south_asia_nocs)
st.plotly_chart(fig2, use_container_width=True)

# -------- C) Top Medal-Winning Athletes --------
st.markdown("### 👑 Top South Asian Medalists")

top_athletes = (
    medals_df[medals_df['NOC'].isin(south_asia_nocs)]
    .groupby(['Name', 'NOC'])
    .agg({'Medal': 'count'})
    .reset_index()
    .rename(columns={'Medal': 'Total Medals'})
    .sort_values(by='Total Medals', ascending=False)
    .head(10)
)

top_athletes['Country'] = top_athletes['NOC'].map(country_labels)
st.dataframe(top_athletes[['Name', 'Country', 'Total Medals']], use_container_width=True)

# -------- D) Dominant Sports by Country --------
st.markdown("### 🏑 Sports Where South Asia Excelled")

sport_medals = (
    medals_df.groupby(['Sport', 'NOC'])
    .size()
    .reset_index(name='Medals')
)

fig3 = px.bar(sport_medals, x='Medals', y='Sport', color='NOC',
              orientation='h', height=400, template='plotly_white')
fig3.for_each_trace(lambda t: t.update(name=country_labels.get(t.name, t.name)))
fig3.update_layout(legend_title_text='Country')
st.plotly_chart(fig3, use_container_width=True)

# -------- E) Summary Table --------
st.markdown("### 📋 South Asia Olympic Summary")

summary_df = sa_df.groupby('NOC').agg(
    Athletes=('ID', 'nunique'),
    Total_Medals=('Medal', lambda x: x.notnull().sum())
).reset_index()
summary_df['Medals per Athlete'] = (summary_df['Total_Medals'] / summary_df['Athletes']).round(2)
summary_df['Country'] = summary_df['NOC'].map(country_labels)

st.dataframe(summary_df[['Country', 'Athletes', 'Total_Medals', 'Medals per Athlete']], use_container_width=True)


# ------------------------ SECTION X: Olympic Fun Facts ------------------------

st.markdown("### 🎉 Olympic Trivia & Fun Facts")

with st.expander("Click to Reveal Amazing Olympic Facts", expanded=False):
    st.markdown("""
- 🧒 **Youngest Medalist Ever**: Just **10 years old** — Dimitrios Loundras, Greece (1896).
- 🧓 **Oldest Medalist Ever**: **72 years old** — Oscar Swahn, Sweden (1920).
- 🥇 **Most Medals by One Athlete**: Michael Phelps with **28 medals** (23 Gold).
- 🏙 **First Modern Olympics**: Held in **Athens, 1896**, only 14 countries participated.
- 🚫 **No Olympics**: Games were cancelled in 1916, 1940, and 1944 due to World Wars.
- 🌍 **Top Medal-Winning Country**: USA with over **2,800 medals**.
    """)

# -------------------- SECTION X: Global Medal Distribution --------------------

st.markdown("### 🌎 Olympic Medal Distribution by Country")

# Filter only medal winners
world_medals = df[df["Medal"].notnull()]

# Group by NOC and count medals
medal_counts = (
    world_medals.groupby("NOC")
    .size()
    .reset_index(name="Total Medals")
)

# Merge with country/team names
noc_country_map = df[["NOC", "Team"]].drop_duplicates()
medal_counts = medal_counts.merge(noc_country_map, on="NOC", how="left")

# Drop duplicates and fix naming issues
medal_counts = medal_counts.drop_duplicates(subset="NOC")

# Plotly Choropleth Map
fig = px.choropleth(
    medal_counts,
    locations="NOC",
    color="Total Medals",
    hover_name="Team",
    color_continuous_scale="plasma",
    projection="natural earth",
    title="🌍 Total Olympic Medals by Country",
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)

=======
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="The Olympic Story: 120 Years of History",
    layout="wide",
    page_icon="🏅"
)


    # 🧾 Your short intro
with st.sidebar.expander("👤 About Hassan"):
    st.image("hassan.png", width=100)
    st.markdown("""
    **Hassan Raza**  
    *Data Analyst*  
    Passionate about uncovering stories hidden in data.  
    
    📫 **Email:** razakhattak123@gmail.com  
    
    📊 **Whatsapp:**  +92335 5102051
    
    🔗 **LinkedIn:** [https://www.linkedin.com/in/hassan-raza-9651b6279)
    
    🔗 **Kaggle:** [https://www.kaggle.com/hassanrazakhattak)
    """)

# 🏅 Title and subtitle on the right
st.title("The Olympic Story: 120 Years of History")
st.markdown("*A deep dive into the athletes, medals, and nations that shaped Olympic history.*")


# 📚 About Dataset in Sidebar
with st.sidebar.expander("📚 About Dataset", expanded=False):
    st.markdown("""
**Olympic History Dataset (1896–2016)**  
_Source: [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)_

This dataset provides a detailed record of Olympic Games history across **both Summer and Winter Games**.

**Key Features:**
- 👤 **Athlete Info**: Name, gender, age, height, weight  
- 🏟️ **Event Info**: Year, season, city, sport, and event  
- 🌍 **Country Details**: Team and NOC codes  
- 🥇 **Medal Records**: Gold, Silver, Bronze (or none)  

**Scale:**
- Over **270,000 athlete entries**
- Covers **120 years** of Olympic history

This rich dataset allows you to explore long-term trends in global athletic participation and medal performance.
""")


##----------------------------- Section 2 FIlters--------------------------------


# Load dataset
df = pd.read_csv("athlete_events.csv")

# Sidebar filters
st.sidebar.header("🎛️ Filter Olympic Data")

# Country filter
country_list = ['All'] + sorted(df['NOC'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country (NOC)", country_list)

# Medal filter
medal_list = ['All', 'Gold', 'Silver', 'Bronze']
selected_medal = st.sidebar.selectbox("Select Medal Type", medal_list)

# Year filter
year_list = ['All'] + sorted(df['Year'].dropna().unique().astype(int).tolist())
selected_year = st.sidebar.selectbox("Select Year", year_list)

# Apply filters
filtered_df = df.copy()

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['NOC'] == selected_country]

if selected_medal != 'All':
    filtered_df = filtered_df[filtered_df['Medal'] == selected_medal]

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == int(selected_year)]



##----------------------------- SECTION 3: Key Metrics Summary (KPIs)--------------------------------


st.markdown("### 📊 Olympic Games Summary")

# Calculate key metrics
total_athletes = filtered_df['ID'].nunique()
total_medals = filtered_df['Medal'].notnull().sum()
total_countries = filtered_df['NOC'].nunique()
total_years = filtered_df['Year'].nunique()

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Athletes", f"{total_athletes:,}")
col2.metric("🥇 Total Medals Awarded", f"{total_medals:,}")
col3.metric("🌍 Participating Countries", f"{total_countries}")
col4.metric("📅 Olympic Years", f"{total_years}")


# ----------------------------- SECTION 4: Visual Insights -----------------------------

st.markdown("### Visual Insights: Olympic Medals Summary")

# Filter medal winners only
medal_winners = filtered_df[filtered_df['Medal'].notnull()]

# Create 3 equal columns for charts
col1, col2, col3 = st.columns(3)

if selected_country == 'All':
    # Top 10 medal-winning countries — shown in col1
    with col1:
        st.markdown("🏅 Top 10 Medal-Winning Countries")

        top_countries = (
            medal_winners.groupby('NOC')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = top_countries.plot(kind='barh', color='gold', ax=ax)

        ax.set_xlabel("Total Medals", fontsize=9)
        ax.set_ylabel("Country (NOC)", fontsize=9)
        ax.set_title("", fontsize=0)
        ax.invert_yaxis()  # So top country shows on top

        # Add value labels
        for p in bars.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{int(width)}',
                            (width + 1, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=9, weight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    # Specific country breakdown — shown in col1
    with col1:
        st.markdown(f"🏅 Medal Breakdown for {selected_country}")

        country_medals = (
            medal_winners[medal_winners['NOC'] == selected_country]
            .groupby('Medal')
            .size()
            .reindex(['Gold', 'Silver', 'Bronze'], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32']
        bars = country_medals.plot(kind='barh', color=colors, ax=ax)

        ax.set_xlabel("Number of Medals", fontsize=9)
        ax.set_ylabel("")
        ax.set_title("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for p in bars.patches:
            width = p.get_width()
            if width > 0:
                ax.annotate(f'{int(width)}',
                            (width + 0.5, p.get_y() + p.get_height() / 2),
                            ha='left', va='center', fontsize=9, weight='bold')

        plt.tight_layout()
        st.pyplot(fig)


# col2-------------------------------

with col2:
    st.markdown("📈 Athlete Participation Over Time")

    # Filter data for selected country (or all)
    if selected_country != 'All':
        country_df = filtered_df[filtered_df['NOC'] == selected_country]
    else:
        country_df = filtered_df.copy()

    # Group by Year and Gender
    athlete_counts = (
        country_df.drop_duplicates(subset=['ID', 'Year'])  # Avoid duplicate athlete counts
        .groupby(['Year', 'Sex'])
        .size()
        .unstack(fill_value=0)
        .rename(columns={'M': 'Male', 'F': 'Female'})
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    athlete_counts.plot(ax=ax, marker='o', linewidth=2)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Number of Athletes", fontsize=9)

    title_label = f"Athletes from {selected_country}" if selected_country != 'All' else "All Athletes"
    ax.set_title(title_label, fontsize=10, fontweight='bold')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title="Gender", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)


 #col3-----------------------------
 
with col3:
    st.markdown("🧑‍🤝‍🧑 Gender Distribution")

    if selected_country != 'All':
        gender_df = filtered_df[filtered_df['NOC'] == selected_country]
    else:
        gender_df = filtered_df.copy()

    gender_counts = (
        gender_df.drop_duplicates(subset=['ID', 'Year'])  # Avoid double counts
        .groupby('Sex')
        .size()
        .reindex(['M', 'F'], fill_value=0)
        .rename(index={'M': 'Male', 'F': 'Female'})
    )

    fig, ax = plt.subplots(figsize=(5, 2.6))
    colors = ['skyblue', 'lightcoral']
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors, textprops={'fontsize': 9})
    ax.axis('equal')
    ax.set_title(f"Gender Ratio ({selected_country})", fontsize=9)
    st.pyplot(fig)


# ----------------------------- SECTION 5: Top Medalists by Gender -----------------------------

st.markdown("### 🧑‍🤝‍🧑 Top Medalists by Gender")

# Confirm gender values in dataset (case-sensitive)
gender_choice = st.selectbox("Select Gender", ['M', 'F'])

# Filter: only medal winners of selected gender
gender_data = filtered_df[
    (filtered_df['Sex'] == gender_choice) &
    (filtered_df['Medal'].notnull())
]

# Apply country filter
if selected_country != 'All':
    gender_data = gender_data[gender_data['NOC'] == selected_country]

# Group by athlete name and country
top_gender_medalists = (
    gender_data.groupby(['Name', 'NOC'])
    .size()
    .reset_index(name='Medal Count')
    .sort_values(by='Medal Count', ascending=False)
    .head(10)
)

# Create label: Name (NOC)
top_gender_medalists['Athlete'] = top_gender_medalists['Name'] + " (" + top_gender_medalists['NOC'] + ")"

# Plot with Plotly
import plotly.express as px

if not top_gender_medalists.empty:
    fig = px.bar(
        top_gender_medalists,
        x='Medal Count',
        y='Athlete',
        orientation='h',
        text='Medal Count',
        color='Medal Count',
        color_continuous_scale='Blues' if gender_choice == 'M' else 'Pinkyl',
        title=f"Top 10 {'Male' if gender_choice == 'M' else 'Female'} Medalists in {selected_country if selected_country != 'All' else 'All Countries'}"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Number of Medals",
        yaxis_title="Athlete (Country)",
        height=450,
        margin=dict(t=50, l=100, r=20, b=40)
    )

    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No {'Male' if gender_choice == 'M' else 'Female'} medalists found for {selected_country}")


# ----------------------------- SECTION 6: Sport Dominance with Country Hover -----------------------------

st.markdown("### 🌍 Sports by Country Dominance")

# Filter medal winners only
sports_df = filtered_df[filtered_df['Medal'].notnull()]

# If specific country selected, keep it filtered
if selected_country != 'All':
    sports_df = sports_df[sports_df['NOC'] == selected_country]

# Group by Sport and Country (NOC)
sport_country = (
    sports_df.groupby(['Sport', 'NOC'])
    .size()
    .reset_index(name='Medal Count')
    .sort_values(by='Medal Count', ascending=False)
)

import plotly.express as px

if not sport_country.empty:
    fig = px.treemap(
        sport_country,
        path=['Sport', 'NOC'],
        values='Medal Count',
        color='Medal Count',
        color_continuous_scale='thermal',
        title=f"🥇 Most Dominant Countries by Sport ({selected_country if selected_country != 'All' else 'All Countries'})",
        hover_data={'Medal Count': True, 'Sport': True, 'NOC': True}
    )

    fig.update_layout(
        margin=dict(t=40, l=20, r=20, b=20),
        height=500
    )

    # 🎯 Center using Streamlit layout
    left, center, right = st.columns([1, 4, 1])

    with center:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No data available for selected filters.")


# ----------------------------- SECTION 7: Athlete Records & Extremes -----------------------------

st.markdown("### 🧍‍♂️ Athlete Profile Highlights")

# ➕ Add Gender Filter
gender_options = ['All'] + list(filtered_df['Sex'].dropna().unique())
selected_sex = st.selectbox("Filter by Gender", gender_options, index=0)

# 🔍 Apply Gender Filter
gender_df = filtered_df.copy()
if selected_sex != 'All':
    gender_df = gender_df[gender_df['Sex'] == selected_sex]

# Clean and deduplicate
valid_df = gender_df.drop_duplicates(subset=["ID", "Name", "Sex", "Age", "Height", "Weight", "Team", "NOC", "Event", "Medal"])
valid_df = valid_df.dropna(subset=["Age", "Height", "Weight"])

# Format text with compact styling
def format_athlete_row(row):
    medal = row["Medal"] if pd.notna(row["Medal"]) else "No Medal"
    return f"""
    <div style="font-size:13px; line-height:1.4">
    <strong>{row['Name']}</strong><br>
    <b>Country:</b> {row['NOC']}<br>
    <b>Event:</b> <i>{row['Event']}</i><br>
    <b>Age:</b> {int(row['Age'])} | <b>Height:</b> {int(row['Height'])} cm | <b>Weight:</b> {int(row['Weight'])} kg<br>
    <b>Medal:</b> {medal}
    </div>
    """

# Find records
if not valid_df.empty:
    heaviest = valid_df.loc[valid_df['Weight'].idxmax()]
    tallest = valid_df.loc[valid_df['Height'].idxmax()]
    shortest = valid_df.loc[valid_df['Height'].idxmin()]
    youngest = valid_df.loc[valid_df['Age'].idxmin()]
    oldest = valid_df.loc[valid_df['Age'].idxmax()]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("#### 🏋️ Heaviest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(heaviest), unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📏 Tallest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(tallest), unsafe_allow_html=True)

    with col3:
        st.markdown("#### 🧍‍♂️ Shortest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(shortest), unsafe_allow_html=True)

    with col4:
        st.markdown("#### 👶 Youngest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(youngest), unsafe_allow_html=True)

    with col5:
        st.markdown("#### 👴 Oldest", unsafe_allow_html=True)
        st.markdown(format_athlete_row(oldest), unsafe_allow_html=True)
else:
    st.warning("No athletes found for the selected gender.")



# ----------------------------- SECTION 8: Most Successful Athletes -----------------------------

st.markdown("### 🏆 Most Successful Athletes")

# 👤 Gender Filter (with unique key to avoid duplicate ID error)
sex_options = ['All'] + list(filtered_df['Sex'].dropna().unique())
selected_sex = st.selectbox("Filter by Gender", sex_options, key="gender_filter_section8")

# 🥇 Step 1: Filter only medal winners
top_athletes_df = filtered_df[filtered_df["Medal"].notnull()].copy()

# 🧍‍♂️ Step 2: Apply gender filter
if selected_sex != 'All':
    top_athletes_df = top_athletes_df[top_athletes_df['Sex'] == selected_sex]

# 🧮 Step 3: Count medals per athlete
athlete_medal_summary = (
    top_athletes_df.groupby(["Name", "NOC"])
    .Medal.value_counts()
    .unstack(fill_value=0)
    .assign(Total=lambda df: df.sum(axis=1))
    .sort_values(by="Total", ascending=False)
    .reset_index()
)

# 📊 Step 4: Pick top 3
top3_athletes = athlete_medal_summary.head(3)

# 📝 Step 5: Card formatting utility
def format_athlete_card(row):
    return f"""
    <div style="font-size:14px; line-height:1.5; padding: 6px 10px; border-radius: 10px; ">
        <strong style="font-size:16px;">🏅 {row['Name']}</strong><br>
        <b>Country:</b> {row['NOC']}<br>
        <b>Gold:</b> 🥇 {row.get('Gold', 0)} |
        <b>Silver:</b> 🥈 {row.get('Silver', 0)} |
        <b>Bronze:</b> 🥉 {row.get('Bronze', 0)}<br>
        <b>Total:</b> <span style="color:green; font-weight:bold;">{row['Total']}</span>
    </div>
    """

# 🧱 Step 6: Layout — 3 equal columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🥇 1st", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[0]), unsafe_allow_html=True)

with col2:
    st.markdown("#### 🥈 2nd", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[1]), unsafe_allow_html=True)

with col3:
    st.markdown("#### 🥉 3rd", unsafe_allow_html=True)
    st.markdown(format_athlete_card(top3_athletes.iloc[2]), unsafe_allow_html=True)



# ---------------------------- SECTION 9: SOUTH ASIA ANALYSIS ----------------------------


st.markdown("### 🌏 South Asia Olympic Analysis")
st.markdown("""
This section explores the Olympic journey of **South Asian countries**:  
🇮🇳 India, 🇵🇰 Pakistan, 🇧🇩 Bangladesh, 🇳🇵 Nepal, 🇱🇰 Sri Lanka, 🇧🇹 Bhutan, 🇲🇻 Maldives  
""")

# Country Codes and Labels
south_asia_nocs = ['IND', 'PAK', 'BAN', 'NEP', 'SRI', 'BHU', 'MDV']
country_labels = {
    'IND': 'India 🇮🇳', 'PAK': 'Pakistan 🇵🇰', 'BAN': 'Bangladesh 🇧🇩',
    'NEP': 'Nepal 🇳🇵', 'SRI': 'Sri Lanka 🇱🇰', 'BHU': 'Bhutan 🇧🇹', 'MDV': 'Maldives 🇲🇻'
}

sa_df = df[df['NOC'].isin(south_asia_nocs)]

# -------- A) Year-wise Participation --------
st.markdown("### 👟 Year-wise Athlete Participation")

participation = sa_df.groupby(['Year', 'NOC'])['ID'].nunique().reset_index(name='Athletes')
fig1 = px.line(participation, x='Year', y='Athletes', color='NOC',
               markers=True, template='plotly_white',
               labels={'NOC': 'Country'}, height=350)
fig1.for_each_trace(lambda t: t.update(name=country_labels.get(t.name, t.name)))
st.plotly_chart(fig1, use_container_width=True)

# -------- B) Total Medal Breakdown --------
st.markdown("### 🥇 Total Medals by Country")

medals_df = sa_df[sa_df['Medal'].notnull()]
medal_summary = medals_df.groupby(['NOC', 'Medal']).size().reset_index(name='Count')

fig2 = px.bar(medal_summary, x='NOC', y='Count', color='Medal',
              barmode='group',
              color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'},
              labels={'NOC': 'Country'}, height=350, template='plotly_white')
fig2.update_xaxes(ticktext=[country_labels[noc] for noc in south_asia_nocs], tickvals=south_asia_nocs)
st.plotly_chart(fig2, use_container_width=True)

# -------- C) Top Medal-Winning Athletes --------
st.markdown("### 👑 Top South Asian Medalists")

top_athletes = (
    medals_df[medals_df['NOC'].isin(south_asia_nocs)]
    .groupby(['Name', 'NOC'])
    .agg({'Medal': 'count'})
    .reset_index()
    .rename(columns={'Medal': 'Total Medals'})
    .sort_values(by='Total Medals', ascending=False)
    .head(10)
)

top_athletes['Country'] = top_athletes['NOC'].map(country_labels)
st.dataframe(top_athletes[['Name', 'Country', 'Total Medals']], use_container_width=True)

# -------- D) Dominant Sports by Country --------
st.markdown("### 🏑 Sports Where South Asia Excelled")

sport_medals = (
    medals_df.groupby(['Sport', 'NOC'])
    .size()
    .reset_index(name='Medals')
)

fig3 = px.bar(sport_medals, x='Medals', y='Sport', color='NOC',
              orientation='h', height=400, template='plotly_white')
fig3.for_each_trace(lambda t: t.update(name=country_labels.get(t.name, t.name)))
fig3.update_layout(legend_title_text='Country')
st.plotly_chart(fig3, use_container_width=True)

# -------- E) Summary Table --------
st.markdown("### 📋 South Asia Olympic Summary")

summary_df = sa_df.groupby('NOC').agg(
    Athletes=('ID', 'nunique'),
    Total_Medals=('Medal', lambda x: x.notnull().sum())
).reset_index()
summary_df['Medals per Athlete'] = (summary_df['Total_Medals'] / summary_df['Athletes']).round(2)
summary_df['Country'] = summary_df['NOC'].map(country_labels)

st.dataframe(summary_df[['Country', 'Athletes', 'Total_Medals', 'Medals per Athlete']], use_container_width=True)


# ------------------------ SECTION X: Olympic Fun Facts ------------------------

st.markdown("### 🎉 Olympic Trivia & Fun Facts")

with st.expander("Click to Reveal Amazing Olympic Facts", expanded=False):
    st.markdown("""
- 🧒 **Youngest Medalist Ever**: Just **10 years old** — Dimitrios Loundras, Greece (1896).
- 🧓 **Oldest Medalist Ever**: **72 years old** — Oscar Swahn, Sweden (1920).
- 🥇 **Most Medals by One Athlete**: Michael Phelps with **28 medals** (23 Gold).
- 🏙 **First Modern Olympics**: Held in **Athens, 1896**, only 14 countries participated.
- 🚫 **No Olympics**: Games were cancelled in 1916, 1940, and 1944 due to World Wars.
- 🌍 **Top Medal-Winning Country**: USA with over **2,800 medals**.
    """)

# -------------------- SECTION X: Global Medal Distribution --------------------

st.markdown("### 🌎 Olympic Medal Distribution by Country")

# Filter only medal winners
world_medals = df[df["Medal"].notnull()]

# Group by NOC and count medals
medal_counts = (
    world_medals.groupby("NOC")
    .size()
    .reset_index(name="Total Medals")
)

# Merge with country/team names
noc_country_map = df[["NOC", "Team"]].drop_duplicates()
medal_counts = medal_counts.merge(noc_country_map, on="NOC", how="left")

# Drop duplicates and fix naming issues
medal_counts = medal_counts.drop_duplicates(subset="NOC")

# Plotly Choropleth Map
fig = px.choropleth(
    medal_counts,
    locations="NOC",
    color="Total Medals",
    hover_name="Team",
    color_continuous_scale="plasma",
    projection="natural earth",
    title="🌍 Total Olympic Medals by Country",
)

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)

>>>>>>> 691e9c8 (Upload full project with dataset)
