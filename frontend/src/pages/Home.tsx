import { useRef, SyntheticEvent } from 'react';
import styled from 'styled-components';
import ArrowDropDown from '@material-ui/icons/ArrowDropDown';

const Box1 = styled.div`
  width: 400px;
  height: 400px;
  background-color: red;
`;

// styles
const Wrapper = styled.main`
  height: 600vh;
  overflow-x: hidden;
`;

const Main = styled.section`
  top: 0;
  width: 100%;
  height: 100vh;
  position: absolute;
  background-repeat: no-repeat;
  background-size: cover;
`;

// 메인페이지를 absolute로 해놓으니 section1도 딸려와서
// section1을 막아줄 임시 박스
const Box = styled.section`
  width: 100%;
  height: 100vh;
  position: relative;
  z-index: -1;
`;

const SectionOne = styled.section`
  width: 100%;
  height: 100vh;
  position: relative;
  border: 1px solid white;
  h1 {
    color: #fff;
  }
`;
const SectionTwo = styled.section`
  width: 100%;
  height: 100vh;
  position: relative;
  border: 1px solid white;
  h1 {
    color: #fff;
  }
`;

const SectionThree = styled.section`
  width: 100%;
  height: 100vh;
  position: relative;
  border: 1px solid white;
  h1 {
    color: #fff;
  }
`;

const ButtonWrapper = styled.div`
  bottom: 0;
  width: 100%;
  height: 56px;
  position: absolute;
  background-color: #222222;
`;

const SlideButton = styled.button`
  margin: 0 auto;
  display: flex;
  height: inherit;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: inherit;
  color: #b2b2b2;
  border: none;

  &:hover {
    color: #f2f2f2;
    transition: all 0.1s ease-in;
  }
`;

// logic
const Home = () => {
  // scroll 이벤트를 위해 useRef로 DOM을 가져옵니다.
  const wrapper = useRef<HTMLElement | null>(null);
  const main = useRef<HTMLElement | null>(null);
  const sectionOne = useRef<HTMLElement | null>(null);
  const sectionTwo = useRef<HTMLElement | null>(null);
  const sectionThree = useRef<HTMLElement | null>(null);

  const handleClick = (e: SyntheticEvent) => {
    // console.log(e.currentTarget.parentElement?.clientHeight);
    const sectionHeight = sectionOne.current?.clientHeight;

    window.scrollTo(
      sectionHeight ? { top: sectionHeight, behavior: 'smooth' } : { top: 0 },
    );
  };

  return (
    <Wrapper ref={wrapper}>
      <Main ref={main}>
        <Box1></Box1>
        <Box1></Box1>
        <ButtonWrapper>
          <SlideButton onClick={handleClick}>
            <span>분석결과 확인하기</span>
            <ArrowDropDown />
          </SlideButton>
        </ButtonWrapper>
      </Main>
      <Box />
      <SectionOne ref={sectionOne}>
        <h1>section_1</h1>
      </SectionOne>
      <SectionTwo ref={sectionTwo}>
        <h1>section_2</h1>
      </SectionTwo>
      <SectionThree ref={sectionThree}>
        <h1>section_3</h1>
      </SectionThree>
    </Wrapper>
  );
};

export default Home;
